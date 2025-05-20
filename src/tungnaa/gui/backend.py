from typing import Optional, Callable, Union, Tuple, List
from numbers import Number
from enum import Enum, Flag
import re
from collections import defaultdict
import cProfile, pstats, io

import threading, os
from threading import Thread, RLock
from queue import Queue
import time
from contextlib import contextmanager
# from fractions import Fraction

import fire

# import numpy as np
# import numpy.typing as npt
import torch
from torch import Tensor
torch.set_num_threads(1)

import sounddevice as sd

from tungnaa import TacotronDecoder, lev

OutMode = Flag('OutMode', ['SYNTH_AUDIO', 'LATENT_AUDIO', 'LATENT_OSC'])
StepMode = Enum('StepMode', ['PAUSE', 'SAMPLER', 'GENERATION'])

def print_audio_devices():
    devicelist = sd.query_devices()
    for dev in devicelist:
        print(f"{dev['index']}: '{dev['name']}' {dev['hostapi']} (I/O {dev['max_input_channels']}/{dev['max_input_channels']}) (SR: {dev['default_samplerate']})")
    return devicelist

class Profiler:
    def __init__(self, enabled):
        self.enabled = enabled
        self.stack = []
        self.mean_wall_ms = defaultdict(float)
        self.mean_proc_ms = defaultdict(float)
        self.n = defaultdict(int)
    @contextmanager
    def __call__(self, label, detail=False):
        if self.enabled:
            if detail:
                pr = cProfile.Profile()
                pr.enable()
                # pr = torch.profiler.profile()
                # pr.__enter__()
            self.stack.append((time.time_ns(), time.process_time_ns()))
        yield None
        if self.enabled:
            t_wall, t_proc = self.stack.pop()
            if detail:
                # pr.__exit__(None, None, None)
                pr.disable()
            proc_ms = (time.process_time_ns() - t_proc)*1e-6
            wall_ms = (time.time_ns() - t_wall)*1e-6
            self.n[label] += 1
            self.mean_wall_ms[label] = (self.mean_wall_ms[label]*(self.n[label]-1) + wall_ms) / self.n[label]
            self.mean_proc_ms[label] = (self.mean_proc_ms[label]*(self.n[label]-1) + proc_ms) / self.n[label]
            # print(
            #     'average', ' '*len(self.stack), label, 
            #     f'wall: {int(self.mean_wall_ms[label])} ms,',
            #     f'cpu: {int(self.mean_proc_ms[label])} ms')
            if self.n[label] > 10 and wall_ms > 2*self.mean_wall_ms[label]:
                print(
                    'slower than 2x average:',
                    '  '*len(self.stack), 
                    label, 
                    f'wall: {int(wall_ms)} ms', 
                    f'cpu: {int(proc_ms)} ms', 
                    f'(average wall {int(self.mean_wall_ms[label])} ms, cpu {int(self.mean_proc_ms[label])} ms)')
                if detail:
                    # print(pr.key_averages().table(
                        # sort_by="self_cpu_time_total", row_limit=4))
                    s = io.StringIO()
                    ps = pstats.Stats(pr, stream=s)
                    ps.sort_stats('tottime')
                    ps.print_stats(4)
                    print(s.getvalue())

class Utterance(list):
    def __init__(self, text):
        self.text = text


class Backend:
    def __init__(self,
        checkpoint:str,
        rave_path:str|None=None,
        audio_in:str|None=None,
        audio_out:str|int=None,
        audio_block:int|None=None,
        audio_channels:int|None=None,
        # sample_rate:int=None,
        synth_audio:bool|None=None,
        latent_audio:bool|None=None,
        latent_osc:bool|None=None,
        osc_sender=None,
        buffer_frames:int=1,
        profile:bool=True,
        jit:bool=False,
        max_model_state_storage:int=512,
        ):
        """
        Args:
            checkpoint: path to tungnaa checkpoint file
            rave_path: path to rave vocoder to use python sound
            audio_in: sounddevice input name/index if using python audio
            audio_out: sounddevice output name/index if using python audio
            audio_block: block size if using python audio
            synth_audio: if True, vocode in python and send stereo audio
            latent_audio: if True, pack latents into a mono audio channel
            latent_osc: if True, send latents over OSC
            buffer_frames: process ahead this many model steps
            profile: if True, print performance profiling info
            jit: if True, compile tungnaa model with torchscript
        """
        self.jit = jit
        self.profile = Profiler(profile)
        # default output modes
        if synth_audio is None:
            synth_audio = rave_path is not None
        if latent_audio is None:
            latent_audio = rave_path is None
        if latent_osc is None:
            latent_osc = False

        self.reset_values = {}

        out_mode = OutMode(0)
        if synth_audio: out_mode |= OutMode.SYNTH_AUDIO
        if latent_audio: out_mode |= OutMode.LATENT_AUDIO
        if latent_osc: out_mode |= OutMode.LATENT_OSC
        self.out_mode = out_mode
        print(f'{self.out_mode=}')

        if audio_channels is None:
            audio_channels = 0
            if OutMode.SYNTH_AUDIO in self.out_mode:
                self.synth_channels = (0,1)
                audio_channels += 2
            else:
                self.synth_channels = tuple()
            if OutMode.LATENT_AUDIO in self.out_mode:
                self.latent_channels = (audio_channels,)
                audio_channels += 1
            else:
                self.latent_channels = tuple()
        else:
            raise NotImplementedError(
                "setting audio_channels not currently supported")
        print(f'{audio_channels=}')

        self.step_mode = StepMode.PAUSE
        self.generate_stop_at_end = False
        self.sampler_stop_at_end = False

        self.latent_biases = []

        # self.temperature = 0.5
        self.temperature = 1.

        # generation
        self.gen_loop_start = 0
        self.gen_loop_end = None
        # sampler
        self.sampler_loop_start = 0
        self.sampler_loop_end = None
        # self.sampler_utterance = -1
        self.sampler_step = 0

        self.osc_sender = osc_sender

        self.frontend_conn = None

        self.max_model_state_storage = max_model_state_storage

        ### move heavy init out of __init__, so it only runs in child process
        ### (Backend.run is called by Proxy._run)
        self.init_args = (buffer_frames, audio_out, checkpoint, rave_path, audio_block, audio_channels)

        # call this from both __init__ and run
        # torch.multiprocessing.set_sharing_strategy('file_system')


    def run(self, conn):
        """run method expected by Proxy"""
        # call this from both __init__ and run
        # torch.multiprocessing.set_sharing_strategy('file_system')

        self.frontend_conn = conn
        # print(f'{self.frontend_conn=} {threading.get_native_id()=} {os.getpid()=}')

        buffer_frames, audio_out, checkpoint, rave_path, audio_block, audio_channels = self.init_args

        self.load_tts_model(checkpoint=checkpoint)

        ### synthesis in python:
        # load RAVE model
        self.load_vocoder_model(rave_path=rave_path)

        ### audio output:
        # make sounddevice stream
        if self.out_mode:
            devicelist = sd.query_devices()
            
            # None throws an error on linux, on mac it uses the default device
            # Let's make this behavior explicit.
            if audio_out is None:
                audio_out = sd.default.device[1]

            audio_device = None
            for dev in devicelist:
                if audio_out in [dev['index'], dev['name']]:
                    audio_device = dev
                    # audio_device_sr = dev['default_samplerate']
                print(f"{dev['index']}: '{dev['name']}' {dev['hostapi']} (I/O {dev['max_input_channels']}/{dev['max_input_channels']}) (SR: {dev['default_samplerate']})")

            # this should not be an error
            # None uses the default device on macOS
            # if audio_device is None:
                # raise RuntimeError(f"Audio device '{audio_out}' does not exist.")

            print(f"USING AUDIO OUTPUT DEVICE {audio_out}:{audio_device}")

            self.active_frame:torch.Tensor = None 
            self.future_frame:Thread = None
            self.frame_counter:int = 0

            sd.default.device = audio_out
            if self.rave_sr:
                if audio_device and audio_device['default_samplerate'] != self.rave_sr:
                    # this should not be an error. On OSX/CoreAudio you can set the device sample rate to the model sample rate.
                    # however on Linux/JACK this throws a fatal error and stops program execution, requiring you to restart Jack to change the sampling rate

                    # TODO: also check RAVE block size vs. audio device block size if possible
                    print("\n------------------------------------");
                    print(f"WARNING: Device default sample rate ({audio_device['default_samplerate']}) and RAVE model sample rate ({self.rave_sr}) mismatch! You may need to change device sample rate manually on some platforms.")
                    print("------------------------------------\n");

                sd.default.samplerate = self.rave_sr # could cause an error if device uses a different sr from model
                print(f"RAVE SAMPLING RATE: {self.rave_sr}")
                print(f"DEVICE SAMPLING RATE: {audio_device['default_samplerate']}")

            # TODO: Tungnaa only uses audio output. Shouldn't we always be using sd.OutputStream?
            try:
                assert len(audio_out)==2
                stream_cls = sd.Stream
            except Exception:
                stream_cls = sd.OutputStream

            self.stream = stream_cls(
                callback=self.audio_callback,
                samplerate=self.rave_sr, 
                blocksize=audio_block, 
                #device=(audio_in, audio_out)
                device=audio_out,
                channels=audio_channels
            )
            
            if self.rave_sr:
                assert self.stream.samplerate == self.rave_sr, f"""
                failed to set sample rate to {self.rave_sr} from sounddevice
                """
        else:
            self.stream = None

        self.text = None
        self.text_rep = None
        self.align_params = None
        self.momentary_align_params = None
        self.step_to_set = None

        self.one_shot_paint = False
        self.latent_feedback = False

        self.lock = RLock()
        self.text_update_thread = None

        # self.frontend_q = Queue()
        self.audio_q = Queue(buffer_frames)     
        self.trigger_q = Queue(buffer_frames)     

        self.states = []

        self.needs_reset = False

        self.step_thread = Thread(target=self.step_loop, daemon=True)
        self.step_thread.start()
 
    def step_loop(self):
        """model stepping thread
        
        for each timestamp received in trigger_q
        send audio frames in audio_q
        """
        while True:
            t = self.trigger_q.get()
            with self.profile('step'):
                frame = self.step(t)
            if frame is not None:
                # with self.profile('frame.numpy'):
                frame = frame.numpy()
            self.audio_q.put(frame)
    
    def load_tts_model(self, checkpoint):
        """helper for loading TTS model, called on initialization but can also be called from GUI"""
        self.model = TacotronDecoder.from_checkpoint(checkpoint)

        # not scripting text encoder for now
        # if self.model.text_encoder is None:
            # self.text_model = TextEncoder()
        # else:
        self.text_model = self.model.text_encoder
        self.model.text_encoder = None

        # def _debug(m):
        #     # print({(k,type(v)) for k,v in m.__dict__.items()})
        #     for k,v in m.__dict__.items():
        #         # print('ATTR', k)
        #         if 'tensor' in str(type(v)).lower(): 
        #             print('TENSOR', k)
        #     for m_ in m.modules():
        #         if m_ != m:
        #             # print('MODULE', m_)
        #             _debug(m_)
        # _debug(self.model)

        if self.jit:
            for m in self.model.modules():
                if hasattr(m, 'parametrizations'):
                    torch.nn.utils.parametrize.remove_parametrizations(
                        m,'weight')
            self.model = torch.jit.script(self.model)
        self.model.eval()
        # self.model.train()

        print(f'{self.num_latents=}')

        # print(f'{self.model.frame_channels=}')
        self.use_pitch = hasattr(self.model, 'pitch_xform') and self.model.pitch_xform

    def load_vocoder_model(self, rave_path):
        """helper for loading RAVE vocoder model, called on initialization but can also be called from GUI"""
        # TODO: The audio engine sampling rate gets set depending on the sampling rate from the vocoder. 
        #   When loading a new vocoder model we need to add some logic to make sure the new vocoder has the same sample rate as the audio system.
        if OutMode.SYNTH_AUDIO in self.out_mode:
            assert rave_path is not None
            self.rave = torch.jit.load(rave_path, map_location='cpu')
            self.rave.eval()
            self.block_size = int(self.rave.decode_params[1])
            try:
                self.rave_sr = int(self.rave.sampling_rate)
            except Exception:
                self.rave_sr = int(self.rave.sr)
            with torch.inference_mode():
                # warmup
                if hasattr(self.rave, 'full_latent_size'):
                    latent_size = self.rave.latent_size + int(
                        hasattr(self.rave, 'pitch_encoder'))
                else:
                    latent_size = self.rave.cropped_latent_size
                self.rave.decode(torch.zeros(1,latent_size,1))
        else:
            self.rave = None
            self.rave_sr = None
    
    @property
    def num_latents(self):
        return self.model.frame_channels
    
    def start_stream(self):
        """helper for start/sampler"""
        # if self.out_mode in (OutMode.LATENT_AUDIO, OutMode.LATENT_OSC):
        #     if not self.loop_thread.is_alive():
        #         self.run_thread = True
        #         self.loop_thread.start()
        # if self.out_mode in (OutMode.LATENT_AUDIO, OutMode.SYNTH_AUDIO):
        # if self.out_mode is not None:
        if not self.stream.active:
            self.stream.start()

    def generate(self):
        """
        start autoregressive alignment & latent frame generation
        """
        # if self.step_mode != StepMode.GENERATION:
            # self.needs_reset = True
        self.step_mode = StepMode.GENERATION
        self.start_stream()

    def pause(self):
        """
        pause generation or sampler
        """
        self.step_mode = StepMode.PAUSE

    def sampler(self):
        """
        start sampler mode
        """
        self.step_mode = StepMode.SAMPLER
        self.start_stream()

    def reset(self):
        """
        reset the model state and alignments history
        """
        self.needs_reset = True

    def cleanup(self):
        """
        Cleanup any resources
        """
        self.step_mode = StepMode.PAUSE
        self.run_thread = False
        # should probably do this more gracefully
        exit(0) # exit the backend process

    def set_text(self, text:str) -> int:
        """
        Compute embeddings for & store a new text, replacing the old text.
        Returns the number of embedding tokens

        Args:
            text: input text as a string

        Returns:
            length of text in tokens
        """
        if (
            self.text_update_thread is not None 
            and self.text_update_thread.is_alive()
        ):
            print('warning: text update still pending')

        # TODO: more general text preprocessing
        text, start, end = self.extract_loop_points(text)
        tokens, text, idx_map = self.text_model.tokenize(text)
        # print(start, end, idx_map)
        # NOTE: positions in text may change fron tokenization (end tokens)
        if start < len(idx_map):
            start = idx_map[start]
        else:
            start = 0
        if end is not None and end < len(idx_map):
            end = idx_map[end]
        else:
            end = None

        # text processing runs in its own thread
        self.text_update_thread = Thread(
            target=self._update_text, 
            args=(text, tokens, start, end), daemon=True)
        self.text_update_thread.start()

        return text

    def extract_loop_points(self, text, tokens='<>'):
        """helper for `set_text`"""
        start_tok, end_tok = tokens
        # TODO: could look for matched brackets, have multiple loops...
        start = text.find(start_tok) # -1 if not found
        # if not len(text):
        #     return text, None, None
        if start < 0:
            start = 0
        else:
            text = text[:start]+text[start+1:]
        end = text.find(end_tok) # -1 if not found
        if end < 0: 
            end = None
        else:
            text = text[:end]+text[end+1:]
            end = max(0, end - 1)
        # print(text, end)
        return text, start, end

    def _update_text(self, text, tokens, start, end):
        """runs in a thread"""
        # store the length of the common prefix between old/new text
        # if self.text is None:
        #     self.prefix_len = 0
        # else:
        #     for i,(a,b) in enumerate(zip(text, self.text)):
        #         print(i, a, b)
        #         if a!=b: break
        #     self.prefix_len = i

        # store a mapping from old text positions to new
        if self.text is None:
            text_index_map = lambda x:x
        else:
            _,_,tm = lev(self.text, text)
            text_index_map = lambda x: tm[x] if x < len(tm) else len(text)-1

        # lock should keep multiple threads from trying to run the text encoder
        # at once in case `input_text` is called rapidly
        with self.lock:
            with torch.inference_mode():
                self.reset_values = dict(
                    text_rep = self.text_model.encode(tokens),
                    text_index_map = text_index_map,
                    text_tokens = tokens,
                    text = text,
                    gen_loop_start = start,
                    gen_loop_end = end,
                )
            self.reset()

    def set_biases(self, biases:List[float]):
        self.latent_biases = biases

    def set_alignment(self, 
            align_params:Optional[Tuple[float, float]]
        ) -> None:
        """
        Send alignment parameters to the backend. If None is passed, alignment painting is off.

        Args:
            align_params: [loc, scale] or None
        """
        self.align_params = align_params
    
    def set_momentary_alignment(self,             
            align_params:Optional[Tuple[float, float]]
        ) -> None:
        """
        Alignment will be set on the next frame only
        """
        self.momentary_align_params = align_params

    def set_state_by_step(self, step):
        self.step_to_set = step

    def set_latent_feedback(self, b:bool):
       self.latent_feedback = b

    def set_generate_stop_at_end(self, b:bool):
        self.generate_stop_at_end = b

    def set_sampler_stop_at_end(self, b:bool):
        self.sampler_stop_at_end = b

    def set_sampler_step(self, step:int):
        self.sampler_step = step % self.total_steps()

    def set_sampler_loop_index(self, 
            start:int=None, end:int=None, 
            utterance:int=None, 
            reset:bool=True):
        """
        Args:
            start: loop start step
            end: loop end step
            utterance: sampler utterance
            reset: if True, immediately go to loop start
        """
        if len(self.states)==0: return

        def wrap(x, n, tag):
            if x is None: return x
            if x < 0: 
                x += n
            if x < 0 or x >= n:
                print(f'warning: out of bounds {tag} {x}')
                x %= n
            return x
        
        if utterance is not None:
            if start is not None:
                start = self.utterance_to_global_step(utterance, start)
            if end is not None:
                end = self.utterance_to_global_step(utterance, end)
        
        if start is None: start = self.sampler_loop_start        
        if end is None: end = self.sampler_loop_end

        # utterance = wrap(utterance, len(self.states), 'utterance')
        start = wrap(start, len(self.states[utterance]), 'loop start')
        end = wrap(end, len(self.states[utterance]), 'loop end')

        # changed = utterance != self.sampler_utterance
        # self.sampler_utterance = utterance

        self.sampler_loop_start = start
        self.sampler_loop_end = end

        # must reset if changing utterance
        if reset:
            self.reset_sampler()

    def set_sampler_loop_text(self, 
            text:str, n:int=-1, 
            start:bool=True, end:bool=True, 
            reset:bool=True):
        """
        Args:
            text: a regular expression string
            n: index of occurrence in history
            start: if True, set the loop start
            end: if True, set the loop end
            reset: if True, immediately go to loop start
        """
        if not (start or end): return
        r = text

        # get all occurences of all matching strings,
        #   and index by order in history

        # first find matches for the regex in utterance texts
        matches = []
        utts = self.states #if n>=0 else reversed(self.states)
        index = 0
        for utt_index, utt in enumerate(utts):
            if len(utt)==0: continue
            # text = utt[0]['text']
            text = utt.text
            for match in re.findall(r, text):
                # error if multiple capture groups in regex
                if not isinstance(match, str):
                    raise ValueError("`set_sampler_loop`: multiple capture groups not supported")
                # add utterance, text index of match
                index = index + text[index:].find(match)
                matches.append((utt_index, index, index+len(match), match))
                # print(f'{utt_index=}, {index=}, {match=}')

        if len(matches)==0:
            print(f'warning: no text matching "{r}" in `set_sampler_loop`')
            return
        
        # for matches, get occurences and flatten into list
        #   simple: occurrence is from entering start to exiting end
        #   TODO better: occurence is when there is a run which
        #       enters the first token / leaves the last token
        #       without any major jumps or reversals
        #   TODO?: allow skips of first / last token?
        occurrences = []
        for u, i, j, m in matches:
            occ_start = None
            utt = self.states[u]
            for k, state in enumerate(utt):
                align = state['align_hard']
                # print(align)

                if align['enter_index']==i:
                    # open/replace occurrence
                    if occ_start is not None:
                        print(f'warning: double open {m=} {occ_start=} {u=} {k=}')
                    occ_start = k
                if align['leave_index']==j:
                    # close occurrence (or ignore)
                    if occ_start is None: 
                        print(f'warning: close without open {m=}')
                        continue
                    occurrences.append((u, occ_start, k))
                    # print(f'{(u, occ_start, k)=}')
                    occ_start = None

        if len(occurrences)==0:
            print(f'warning: no occurrences of {matches} in `set_sampler_loop`')
            return
        
        # index into list using n
        u, loop_start, loop_end = occurrences[n]

        loop_start = self.utterance_to_global_step(u, loop_start)
        loop_end = self.utterance_to_global_step(u, loop_end)
        # TODO: efficient search for index of occurence
                
        # set loop points
        # if utterance changes, unset points which aren't set
        # changed = u != self.sampler_utterance
        # self.sampler_utterance = u

        if start:
            self.sampler_loop_start = loop_start
        # elif changed:
            # self.sampler_loop_start = 0

        if end:
            self.sampler_loop_end = loop_end
        # elif changed:
            # self.sampler_loop_end = None

        # must reset if changing utterance
        # if changed or reset:
        if reset:
            self.reset_sampler()

        # print(f'{self.sampler_utterance=}, {self.sampler_step=}, {self.sampler_loop_start=}, {self.sampler_loop_end=}')
    
            
    def reset_sampler(self):
        self.step_mode = StepMode.SAMPLER
        self.sampler_step = self.sampler_loop_start
        

    def process_frame(self, 
            latent_t:Optional[Tensor]=None, 
            align_t:Optional[Tensor]=None
        ) -> tuple[Tensor, Tensor]:
        """
        Generate an audio(rave latents) and alignment frame.

        Args:
            latent_t: [batch, RAVE latent size]
                last frame of audio feature if using `latent feedback` mode
            align_t: [batch, text length in tokens]
                explicit alignments if using `paint` mode
        Returns:
            latent_t: [batch, RAVE latent size]
                next frame of audio feature
            align_t: [batch, text length in tokens]
                alignments to text
        """
        with torch.inference_mode():
            with self.profile('tts', detail=False):
                r = self.model.step(
                    alignment=align_t, audio_frame=latent_t, 
                    temperature=self.temperature)
                latent_t, align_t, 
        # use low precision for storage
        return r['output'].half(), r['alignment'].half()
    
    def set_temperature(self, t):
        self.temperature = t
    
    def do_audio_block(self, outdata, sdtime):
        """loop over samples of output, requesting frames from the audio thread
        and pulling them from the queue as needed
        """
        outdata[:,:] = 0
        c = self.synth_channels+self.latent_channels
        for i in range(outdata.shape[0]):
            # if the current frame is exhausted, delete it
            if self.active_frame is not None:
                if self.frame_counter >= self.active_frame.shape[-1]:
                    self.active_frame = None
                    self.frame_counter = 0

            # if no active frame, try to get one from step thread
            if self.active_frame is None:
                if not self.audio_q.empty(): 
                    self.active_frame = self.audio_q.get()

            if not self.trigger_q.full():
                # use ADC input time as timestamp
                timestamp = sdtime.inputBufferAdcTime
                self.trigger_q.put(timestamp)

            if self.active_frame is None:
                if self.playing:
                    print(f'audio: dropped frame')
                return
            else:
                # read next audio sample out of active model frame 
                # TODO: batch/multichannel handling
                outdata[i,c] = self.active_frame[:, self.frame_counter]
                self.frame_counter += 1

    def audio_callback(self,*a):
        """sounddevice callback main loop"""
        if len(a)==4: # output device only case
            (
                outdata,#: np.ndarray, #[frames x channels]
                frames, sdtime, status
            ) = a
        elif len(a)==5: # input and output device
            (
                indata, outdata, #np.ndarray, #[frames x channels]
                frames, sdtime, status
            ) = a

        self.do_audio_block(outdata, sdtime)
         
    def utterance_empty(self, utterance=-1):
        try:
            return len(self.states[utterance])==0
        except IndexError:
            return True
        
    def utterance_len(self, utterance=-1):
        return len(self.states[utterance])
    
    def global_step_to_utterance(self, global_step):
        utterance = 0
        step = global_step
        while True:
            if utterance >= len(self.states):
                raise ValueError('warning: global_step out of bounds')
            ul = len(self.states[utterance])
            if step < ul:
                break
            step -= ul
            utterance += 1

        return utterance, step
    
    def utterance_to_global_step(self, utterance, step):
        if utterance >= len(self.states):
            raise ValueError('warning: utterance out of bounds')
        return step + sum(len(self.states[i]) for i in range(utterance-1))

    def prev_state(self, *a, step=-1, utterance=-1, global_step=None):
        """convenient access to previous states"""
        if global_step is not None:
            try:
                utterance, step = self.global_step_to_utterance(global_step)
            except ValueError:
                return None

        s = self.states[utterance][step]
        for k in a:
            s = s[k]
        return s
    
    def total_steps(self):
        return sum(len(s) for s in self.states)

    def hard_alignment(self, align_t):
        """return index and character of hard alignment"""
        i = align_t.argmax().item()
        # print(i)
        c = self.text[i] if i < len(self.text) else None

        i_enter = None
        i_leave = None
        c_enter = None
        c_leave = None
        if not self.utterance_empty():
            # print(self.states)
            prev_align = self.prev_state('align_hard')

            if i>prev_align['index']:
                i_enter = i
                i_leave = prev_align['index']
                c_enter = c
                c_leave = prev_align['char']

        return {
            'index':i, 
            'char':c, 
            'enter_index':i_enter,
            'leave_index':i_leave,
            'enter_char':c_enter,
            'leave_char':c_leave
        }
    
    def do_reset(self):
        """perform reset of model states/text (called from `step`)"""
        print('RESET')
        for k,v in self.reset_values.items():
            setattr(self, k, v)
        
        if self.text_rep is not None:
            self.model.reset(self.text_rep)
            if self.align_params is None:
                # go to loop start after reset
                self.momentary_align_params = (self.gen_loop_start, 1)
            elif not self.utterance_empty():
                # unless painting alignments -- then try to stay
                # in approximately the same spot
                # TODO this doesn't work because the frontend just sets it again based on the slider -- need to add bidirectional control
                self.align_params = (
                    self.text_index_map(round(self.align_params[0])), 1)
                
            # remove old RNN states
            if len(self.states):
                for state in self.states[-1]:
                    self.strip_states(state)

            # start a new utterance
            self.states.append(Utterance(self.text))
            ## for now, only sampler of current utterance is supported
            self.sampler_step = 0
            # in the future, it might be useful to encode texts without yet starting a new utterance. for now, it makes more sense if hitting encode 
            # always starts generation
            self.generate()

            self.needs_reset = False


        return {
            'reset': True, 
            'text': self.text, 
            'num_latents': self.num_latents, 
            'use_pitch': self.use_pitch
            }

    def paint_alignment(self):
        """helper for `step_gen`"""
        align_params = self.momentary_align_params or self.align_params
        self.momentary_align_params = None
        if align_params is None:
            return None
        loc, scale = align_params
        n_tokens = self.text_rep.shape[1]
        loc = max(0, min(n_tokens, loc))
        deltas = torch.arange(n_tokens) - loc
        deltas = deltas / (0.5 + scale) # sharpness modulated by speed 
        # discrete gaussian, sums exactly to 1 over text positions
        logits = -deltas**2
        res = logits.exp() 
        res = res / res.sum()
        return res[None]
    

    def convert_numpy(self, item):
        if hasattr(item, 'numpy'):
            return item.numpy()
        # elif isinstance(item, dict):
            # return {k:self.convert_numpy(v) for k,v in item.items()}
        # elif hasattr(item, '__len__'):
            # return [self.convert_numpy(i) for i in item]
        else:
            return item
    
    def send_state(self, state):   
        state = {
            k:self.convert_numpy(v)
            for k,v in state.items()
            if k!='model_state'
        }
        self.frontend_conn.send(state)

    def step(self, timestamp):
        """compute one vocoder frame of generation or sampler"""
        # if self.frontend_q.qsize() > 100:
            # self.frontend_q.get()
            # print('frontend queue full, dropping')

        state = {'text':self.text}

        # reset text and model states
        if self.needs_reset:
            state |= self.do_reset()

        # print(f'{self.frontend_conn=} {threading.get_native_id()=} {os.getpid()=}')

        if self.text_rep is None or self.step_mode==StepMode.PAUSE:
            if len(state):
                # self.frontend_q.put(state)
                if self.frontend_conn is not None:
                    self.send_state(state)
                else:
                    print('warning: frontend_conn does not exist')
            return None
        if self.step_mode==StepMode.GENERATION:
            # with self.profile('step_gen'):
            state |= self.step_gen(timestamp)
        elif self.step_mode==StepMode.SAMPLER:
            with self.profile('step_sampler'):
                state |= self.step_sampler(timestamp)
        # else:
            # print(f'WARNING: {self.step_mode=} in step')

        if self.osc_sender is not None:
            self.osc_sender(state)

        if len(self.latent_biases):
            with torch.inference_mode():
                bias = torch.tensor(self.latent_biases)[None]
                state['latent_t'] += bias

        # send to frontend
        # self.frontend_q.put(state)
        if self.frontend_conn is not None:
            self.send_state(state)
        else:
            # print(state)
            print('warning: frontend_conn does not exist')

        latent = state['latent_t'] # batch, channel
        batch, n_latent = latent.shape
        assert batch==1

        c_synth, c_latent = len(self.synth_channels), len(self.latent_channels)

        with torch.inference_mode():
            audio_frame = torch.zeros(
                c_synth+c_latent, 
                self.model.block_size) # channel, time
            
            if OutMode.SYNTH_AUDIO in self.out_mode:
                # allow creative use of vocoders with mismatched sizes
                rave_dim = self.rave.decode_params[0]
                common_dim = min(rave_dim, latent.shape[-1])
                latent_to_rave = torch.zeros(latent.shape[0], rave_dim)
                latent_to_rave[:,:common_dim] = latent[:,:common_dim]
                ### run the vocoder
                with self.profile('vocoder'):
                    audio = self.rave.decode(
                        latent_to_rave[...,None])[0] # channel, time
                ###
                audio_frame[:c_synth, :] = audio
                
            if OutMode.LATENT_AUDIO in self.out_mode:
                latent = torch.cat((
                    torch.zeros(batch, 1), # zero to ensure trigger
                    torch.full((batch, 1), n_latent), # trigger, number of latents
                    latent
                    ), dim=1) / 1024 # scale down in case audio gets sent to speakers
                # latent x batch

                audio_frame[c_synth:, :2+n_latent] = latent

        return audio_frame
        # send to audio thread
        # print(f'DEBUG: queuing audio {id(audio)}')
        # self.audio_q.put(audio_frame)

    def step_sampler(self, timestamp):
        """sampler branch of main `step` method"""
        # print(f'{self.sampler_step=}')

        # if self.utterance_empty(utterance=self.sampler_utterance):
        if self.total_steps()==0:
            print('nothing to play back')
            self.step_mode = StepMode.PAUSE
            return None
        
        state = self.prev_state(global_step=self.sampler_step) or {}

        self.sampler_step += 1

        if (
            self.sampler_step == self.sampler_loop_end 
            or self.sampler_step >= self.total_steps()
            ):
            self.sampler_step = self.sampler_loop_start
            if self.sampler_stop_at_end:
                self.step_mode = StepMode.PAUSE


        state = {**state}
        state['latent_t'] = state['latent_t'].clone()
        state['timestamp'] = timestamp
        state['sampler'] = True

        return state
    

    def step_gen(self, timestamp):
        """generation branch of main `step` method"""
        # loop end
        if (
            self.align_params is None 
            and self.gen_loop_end is not None 
            and not self.utterance_empty()
            and self.prev_state('align_hard', 'index') in (
                self.gen_loop_end, self.gen_loop_end+1)
        ):
            self.momentary_align_params = (self.gen_loop_start, 1)

        # set just model states
        # TODO: possibility to set to previous utterance?
        if self.step_to_set is not None:
            step = self.step_to_set
            with torch.inference_mode():
                try:
                    self.model.set_state(
                        self.prev_state('model_state', step=step))
                except Exception:
                    print(f'WARNING: failed to set {step=} ')
                    raise
            self.step_to_set = None

        if self.model.memory is None:
            print('skipping: model not initialized')
            if self.rave is not None:
                self.audio_q.put(None)
            return None

        if not self.utterance_empty() and self.latent_feedback:
            # pass the last vocoder frame back in
            latent_t = self.prev_state('latent_t')
        else:
            latent_t = None

        align_t = self.paint_alignment()

        ##### run the TTS model
        latent_t, align_t = self.process_frame(
            # self.mode, 
            latent_t=latent_t, align_t=align_t)
        #####

        # NOTE
        ### EXPERIMENTAL juice knob
        # latent_t = latent_t * 1.1
        ###

        state = {
            # 'text':self.text,
            'latent_t':latent_t, 
            'align_t':align_t,
            'timestamp':timestamp,
            'model_state':self.model.get_state(),
            'align_hard':self.hard_alignment(align_t)
            }
        
        if (
            self.generate_stop_at_end 
            and state['align_hard']['index']==(len(self.text)-1)
        ):
            self.step_mode = StepMode.PAUSE
            # self.reset()

        utt = self.states[-1]
        utt.append(state)
        # remove RNN states after certain number of steps
        if len(utt) >= self.max_model_state_storage:
            self.strip_states(utt[-self.max_model_state_storage])
        return state
    
    def strip_states(self, state):
        """remove RNN states"""
        state.pop('model_state', None)

    @property
    def playing(self):
        return self.step_mode!=StepMode.PAUSE


def main(checkpoint='../rtalign_004_0100.ckpt'):
    from tungnaa.gui.senders import DirectOSCSender
    b = Backend(checkpoint, osc_sender=DirectOSCSender())
    b.start()
    print('setting text')
    n = b.set_text('hey, hi, hello, test. text. this is a text.') # returns the token length of the text
    print(f'text length is {n} tokens')
    print('setting alignment')
    i = 0
    while True:
        b.set_alignment(torch.randn(1,n).softmax(-1)) # ignored if mode == paint
        if i==300:
            print('setting text')
            n = b.set_text('this is a new text')
        if i==500:
            print('setting mode')
            b.set_mode('paint') # the other available mode is 'infer'

        while not b.q.empty():
            r = b.q.get()
            a = r['align_t'].argmax().item()
            print('█'*(a+1) + ' '*(n-a-1) + '▏')

            print(' '.join(f'{x.item():+0.1f}' for x in r['latent_t'][0]))
            
        time.sleep(1e-2)
        i += 1

if __name__=='__main__':
    fire.Fire(main)
