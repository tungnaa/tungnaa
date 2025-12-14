"""
Backend runs in a Proxy: main thread loops, handling commands from the frontend.
an audio thread is driven by sounddevice callback, requests blocks of audio 
    this gets replaced when changing audio device
a step_thread loops, checking for requested audio blocks and generating them
    this gets replaced when changing models
short-lived threads are also created to async load new models and encode text
"""

# reset can be called at any time, models may not be loaded
# reset assumes models are loaded
# setting text triggers reset
#    but this waits for model to be loaded
# setting model -> set text -> reset
# so just need to ignore resets from frontend before model is loaded

from typing import Optional, Callable, Union, Tuple, List, Dict, Any
from numbers import Number
from enum import Enum, Flag
import re
from collections import defaultdict
import cProfile, pstats, io
import traceback

# import threading, os
from threading import Thread, RLock
from queue import Queue
import time
from contextlib import contextmanager
from pathlib import Path
# from fractions import Fraction

# import fire

# import numpy as np
# import numpy.typing as npt
import torch
from torch import Tensor
import torch.nn.utils.parametrize
torch.set_num_threads(1)

import sounddevice as sd

from tungnaa import TacotronDecoder, lev

OutMode = Flag('OutMode', ['SYNTH_AUDIO', 'LATENT_AUDIO', 'LATENT_OSC'])
StepMode = Enum('StepMode', ['PAUSE', 'SAMPLER', 'GENERATION'])

def print_audio_devices():
    devicelist = sd.query_devices()
    assert isinstance(devicelist, sd.DeviceList) #typing
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
        self.mean_intercall_ns = 0
        self.intercall_decay = 0.95
        self.last_toplevel_call = None
    @contextmanager
    def __call__(self, label, detail=False):
        if self.enabled:
            if detail:
                pr = cProfile.Profile()
                pr.enable()
                # pr = torch.profiler.profile()
                # pr.__enter__()
            t = time.time_ns()
            if not len(self.stack):
                if self.last_toplevel_call is not None:
                    dt = t-self.last_toplevel_call
                    if self.mean_intercall_ns==0:
                        self.mean_intercall_ns = dt
                    else:
                        self.mean_intercall_ns = (
                            self.mean_intercall_ns*self.intercall_decay 
                            + (1-self.intercall_decay) * dt)
                self.last_toplevel_call = t
            self.stack.append((t, time.process_time_ns()))
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
            intercall_ms = self.mean_intercall_ns*1e-6
            if wall_ms > intercall_ms * 0.9:
                color = '\033[31;1m' # red
            elif wall_ms > intercall_ms * 0.25:
                color = '\033[33;1m' # yellow
            else:
                color = '\033[32;1m' # green
            reset = '\033[0m'
            space = '    '*len(self.stack)
            print(
                space, 
                label + ' '*max(1, 16-len(label)-len(space)), 
                color + f'wall: {int(wall_ms)} ms' + reset, 
                f'\tcpu: {int(proc_ms)} ms', 
                # f'\t(mean wall {int(self.mean_wall_ms[label])} ms, \tcpu {int(self.mean_proc_ms[label])} ms)'
                )
            if not len(self.stack):
                print(f'inferred frame budget: {int(intercall_ms)} ms')
                print('-'*40)
            # print(
            #     'average', ' '*len(self.stack), label, 
            #     f'wall: {int(self.mean_wall_ms[label])} ms,',
            #     f'cpu: {int(self.mean_proc_ms[label])} ms')
            # if self.n[label] > 10 and wall_ms > 2*self.mean_wall_ms[label]:
                # print(
                #     'slower than 2x average:',
                #     '  '*len(self.stack), 
                #     label, 
                #     f'wall: {int(wall_ms)} ms', 
                #     f'cpu: {int(proc_ms)} ms', 
                #     f'(average wall {int(self.mean_wall_ms[label])} ms, cpu {int(self.mean_proc_ms[label])} ms)')
                # if detail:
                #     # print(pr.key_averages().table(
                #         # sort_by="self_cpu_time_total", row_limit=4))
                #     s = io.StringIO()
                #     ps = pstats.Stats(pr, stream=s)
                #     ps.sort_stats('tottime')
                #     ps.print_stats(4)
                #     print(s.getvalue())

class Utterance(list):
    def __init__(self, text):
        self.text = text


class Backend:
    def __init__(self,
        audio_block:int|None=None,
        audio_channels:int|None=None,
        # sample_rate:int=None,
        synth_audio:bool=True,
        latent_audio:bool=False,
        latent_osc:bool=False,
        osc_sender=None,
        buffer_frames:int=1,
        profile:bool=True,
        jit:bool=False,
        max_model_state_storage:int=512,
        ):
        """
        Lightweight init which runs in the frontend process.
        Args:
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

        self.reset_values = {}

        out_mode = OutMode(0)
        if synth_audio: out_mode |= OutMode.SYNTH_AUDIO
        if latent_audio: out_mode |= OutMode.LATENT_AUDIO
        if latent_osc: out_mode |= OutMode.LATENT_OSC
        self.out_mode = out_mode
        print(f'{self.out_mode=}')

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

        # self.audio_in = audio_in
        self.audio_out = None
        self.audio_block = audio_block
        self.audio_channels = audio_channels
        self.buffer_frames = buffer_frames
        self.stream = None

        self.vocoder_sr = None
        self.model = None
        self.text_model = None
        self.vocoder = None

        self.step_thread = None
        self.raw_text = ''
        self.text = None

        self.use_pitch = None

        self.needs_vocoder_replace = False
        self.needs_model_replace = False

        # call this from both __init__ and run
        # torch.multiprocessing.set_sharing_strategy('file_system')


    def run(self, conn):
        """
        run method expected by Proxy class.

        initialize things which won't be sent over the Pipe from the parent copy
        """
        # call this from both __init__ and run
        # torch.multiprocessing.set_sharing_strategy('file_system')

        self.frontend_conn = conn
        # print(f'{self.frontend_conn=} {threading.get_native_id()=} {os.getpid()=}')

        # TODO make this a method + stored schema instead of lambda
        self.text_index_map = lambda x:x

        self.text = None
        self.text_rep = None
        self.align_params = None
        self.momentary_align_params = None
        self.step_to_set = None

        self.one_shot_paint = False
        self.latent_feedback = False

        self.lock = RLock()

        self.text_update_thread = None    
        self.tts_model_update_thread = None    
        self.vocoder_update_thread = None    

        self.states = []

        self.needs_reset = False
        self.needs_model_replace = False

        self.audio_q = Queue(self.buffer_frames)     
        self.trigger_q = Queue(self.buffer_frames) 

        self.step_thread = Thread(target=self.step_loop, daemon=True)
        self.step_thread.start()
 
    def step_loop(self):
        """model stepping thread
        
        for each timestamp received in trigger_q
        send audio frames in audio_q
        """
        # while self.stepping:
        while True:
            t = self.trigger_q.get()
            with self.profile('step'):
                frame = self.step(t)
            if frame is not None:
                # with self.profile('frame.numpy'):
                frame = frame.numpy()
            self.audio_q.put(frame)

    def set_audio_device(self, 
            audio_out=None, 
            audio_block=None, 
            audio_channels=None, 
            ):
        print("======= set audio device ======")
        ### audio output:
        # make sounddevice stream

        if audio_out is None: audio_out = self.audio_out
        if audio_block is None: audio_block = self.audio_block
        if audio_channels is None: audio_channels = self.audio_channels

        # print(f'{audio_out=} {audio_block=} {audio_channels=} {buffer_frames=}')
        
        print(f'{self.out_mode=}')
        if self.out_mode:

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

            devicelist = sd.query_devices()
            assert isinstance(devicelist, sd.DeviceList) # typing
            
            audio_device = None
            for dev in devicelist:
                # search by either name or index
                if audio_out in [dev['index'], dev['name']]:
                    audio_device = dev
                    # audio_device_sr = dev['default_samplerate']

            print(f"USING AUDIO OUTPUT DEVICE {audio_out}:{audio_device}")

            self.active_frame:torch.Tensor|None = None 
            self.frame_counter:int = 0

            # sd.default.device = audio_out
            if self.vocoder_sr:
                if audio_device and audio_device['default_samplerate'] != self.vocoder_sr:
                    # TODO: also check RAVE block size vs. audio device block size if possible?
                    print("\n------------------------------------");
                    print(f"WARNING: Device default sample rate ({audio_device['default_samplerate']}) and RAVE model sample rate ({self.vocoder_sr}) mismatch! You may need to change device sample rate manually on some platforms.")
                    print("------------------------------------\n");

                # sd.default.samplerate = self.vocoder_sr # could cause an error if device uses a different sr from model
                # print(f"RAVE SAMPLING RATE: {self.vocoder_sr}")
                # if audio_device is not None:
                #     print(f"DEVICE SAMPLING RATE: {audio_device['default_samplerate']}")

            # revisit this if we ever add audio input
            # try:
            #     assert audio_out is not None and len(audio_out)==2
            #     stream_cls = sd.Stream
            # except Exception:
            # stream_cls = sd.OutputStream

            restart = False
            if self.stream is not None:
                if self.stream.active:
                    restart = True
                print("closing old stream")
                self.stream.close()

            print("creating new stream")
            try:
                # raise sd.PortAudioError("""DEBUG""")
                self.stream = sd.OutputStream(
                    callback=self.audio_callback,
                    samplerate=self.vocoder_sr, 
                    blocksize=audio_block, 
                    device=audio_out,
                    channels=audio_channels
                )
            except sd.PortAudioError:
                # print(audio_device)
                if audio_device is None: raise
                print(f"falling back to {audio_device['default_samplerate']=}")
                self.stream = sd.OutputStream(
                    callback=self.audio_callback,
                    samplerate=audio_device['default_samplerate'], 
                    blocksize=audio_block, 
                    device=audio_out,
                    channels=audio_channels
                )

            print(f'{self.stream.blocksize=} {self.stream.samplerate=}')

            if restart:
                print('starting new stream')
                self.stream.start()
            
            if self.vocoder_sr and self.stream.samplerate != self.vocoder_sr:
                print(f"warning: failed to set sample rate to {self.vocoder_sr} from sounddevice")

            self.audio_out = audio_out
            self.audio_block = audio_block
        else:
            self.stream = None
    
    def set_tts_model(self, checkpoint):
        self.pause()
        self.tts_model_update_thread = Thread(
            target=self._update_tts_model, 
            args=(checkpoint,), daemon=True)
        self.tts_model_update_thread.start()

    def _update_tts_model(self, checkpoint):
        """helper for loading TTS model"""
        print('loading new model')
        model = TacotronDecoder.from_checkpoint(checkpoint)

        # not scripting text encoder for now
        # if self.model.text_encoder is None:
            # self.text_model = TextEncoder()
        # else:
        text_model = model.text_encoder
        model.text_encoder = None

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
            try:
                for m in model.modules():
                    if hasattr(m, 'parametrizations'):
                        torch.nn.utils.parametrize.remove_parametrizations(
                            m,'weight')
                model = torch.jit.script(model)
            except Exception:
                print('warning: jit failed')
        model.eval()

        self.replace_models = (model, text_model)
        self.needs_model_replace = True


    def do_model_replace(self):
        print('replacing model')
        self.model, self.text_model = self.replace_models

        print(f'{self.num_latents=}')
        self.use_pitch = hasattr(self.model, 'pitch_xform') and self.model.pitch_xform

        self.pause()
        self.text_rep = None
        self.needs_model_replace = False

        self.launch_text_update()

        return {}
    
    def set_vocoder(self, checkpoint):
        self.vocoder_update_thread = Thread(
            target=self._update_vocoder, 
            args=(checkpoint,), daemon=True)
        self.vocoder_update_thread.start()

    def _update_vocoder(self, rave_path):
        """helper for loading RAVE vocoder model"""
        # self.kill_step_thread()
        # TODO: The audio engine sampling rate gets set depending on the sampling rate from the vocoder. 
        #   When loading a new vocoder model we need to add some logic to make sure the new vocoder has the same sample rate as the audio system.
        if OutMode.SYNTH_AUDIO in self.out_mode:
            assert rave_path is not None
            vocoder = torch.jit.load(rave_path, map_location='cpu')
            vocoder.eval()
            self.block_size = int(vocoder.decode_params[1])
            try:
                vocoder_sr = int(vocoder.sampling_rate)
            except Exception:
                vocoder_sr = int(vocoder.sr)
            with torch.inference_mode():
                # warmup
                if hasattr(vocoder, 'full_latent_size'):
                    latent_size = vocoder.latent_size + int(
                        hasattr(vocoder, 'pitch_encoder'))
                else:
                    latent_size = vocoder.cropped_latent_size
                vocoder.decode(torch.zeros(1,latent_size,1))
        else:
            vocoder = None
            vocoder_sr = None
            latent_size = None
        self.replace_vocoder = (vocoder, vocoder_sr, latent_size)
        self.needs_vocoder_replace = True

    def do_vocoder_replace(self):
        self.vocoder, self.vocoder_sr, latent_size = self.replace_vocoder
        if self.stream is not None and self.vocoder_sr != self.stream.samplerate:
            self.set_audio_device()
        self.needs_vocoder_replace = False
        return {'vocoder_num_latents': latent_size}
    
    @property
    def num_latents(self):
        """latent dim (of TTS model)"""
        if self.model is None: return None
        return self.model.frame_channels
    
    def start_stream(self):
        """helper for start/sampler"""
        if self.stream is None:
            print("warning: no audio stream")
            return
        if not self.stream.active:
            self.stream.start()

    def generate(self):
        """
        start autoregressive alignment & latent frame generation
        """
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

    def set_text(self, text:str):
        """
        Set a new text and flag it for tokenizing/embedding via the model.
        Compute embeddings for & store a new text, replacing the old text.
        Returns the processed plain text synchronously while running the 
        text embedding model in a thread.

        Args:
            text: input text as a string

        Returns:
            processed text
        """
        if (
            self.text_update_thread is not None 
            and self.text_update_thread.is_alive()
        ):
            print('warning: text update still pending')
        self.raw_text = text
        self.launch_text_update()

    def launch_text_update(self):
        # text processing runs in its own thread
        self.text_update_thread = Thread(
            target=self._update_text, 
            args=(self.raw_text,), daemon=True)
        self.text_update_thread.start()
    
    def _update_text(self, text):
        """runs in a thread"""
        # store the length of the common prefix between old/new text
        # if self.text is None:
        #     self.prefix_len = 0
        # else:
        #     for i,(a,b) in enumerate(zip(text, self.text)):
        #         print(i, a, b)
        #         if a!=b: break
        #     self.prefix_len = i

        # assert self.text_model is not None, "error: no text encoder loaded"
        # wait until text model is set if necessary
        while self.text_model is None:
            time.sleep(1e-2)

         # TODO: more general text preprocessing
        text, start, end = self.extract_loop_points(text)
        if self.text_model is None:
            print("warning: no text encoder loaded")
            return ''
        tokens, text, idx_map = self.text_model.tokenize(text)
        # print(start, end, idx_map)
        # NOTE: positions in text may change from tokenization (end tokens)
        if start < len(idx_map):
            start = idx_map[start]
        else:
            start = 0
        if end is not None and end < len(idx_map):
            end = idx_map[end]
        else:
            end = None

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
                # these are attributes which need to be set when reset occurs
                self.reset_values.update(dict(
                    text_rep = self.text_model.encode(tokens),
                    text_index_map = text_index_map,
                    text_tokens = tokens,
                    text = text,
                    gen_loop_start = start,
                    gen_loop_end = end,
                ))
            # flag for a model reset
            self.reset()

    def extract_loop_points(self, text, tokens='<>'):
        """helper for `_update_text`"""
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
    
    def do_reset(self):
        """perform reset of model states/text (called from `step`)"""
        print('RESET')
        # set arbitrary attributes required to perform a reset
        for k,v in self.reset_values.items():
            setattr(self, k, v)
        
        if self.text_rep is not None:
            assert self.model is not None, "do_reset called while model is None"
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

    def set_biases(self, biases:List[float]):
        self.latent_biases = biases

    def set_alignment(self, 
            align_params:Optional[Tuple[float, float]],
            momentary:bool = False,
            add:bool = False,
        ) -> None:
        """
        Set alignment parameters from the frontend. 
        If None is passed, alignment painting is off.

        Args:
            align_params: [loc, scale] or None
            momentary: if True, set for only the next inference step
            add: if True, increment location instead of setting
        """
        if align_params is None: 
            self.align_params = align_params
            return
        
        loc, scale = align_params
        if add:
            try:
                loc = loc + self.states[-1][-1]['align_hard']['index']
            except Exception:
                traceback.print_exc()
        align_params = loc, scale

        if momentary:
            self.momentary_align_params = align_params
        else:
            self.align_params = align_params

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

        if utterance is not None:
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
        c = self.text[i] if self.text is not None and i < len(self.text) else None

        i_enter = None
        i_leave = None
        c_enter = None
        c_leave = None
        if not self.utterance_empty():
            # print(self.states)
            prev_align = self.prev_state('align_hard')
            if prev_align is None:
                print('error: alignment missing from previous state')
            else:
                if i>prev_align['index']:
                    i_enter = i
                    i_leave = prev_align['index']
                    c_enter = c
                    c_leave = prev_align['char']

        return {
            'index':i, 
            'char':c, # can be None if index out of bounds
            'enter_index':i_enter,
            'leave_index':i_leave,
            'enter_char':c_enter,
            'leave_char':c_leave
        }

    def paint_alignment(self):
        """helper for `step_gen`"""
        align_params = self.momentary_align_params or self.align_params
        # print(f'{align_params=}')
        self.momentary_align_params = None
        if align_params is None:
            return None
        loc, scale = align_params

        if self.text_rep is None:
            print("error: text_rep is None in paint_alignment")
            return None

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
        if self.frontend_conn is not None:
            state = {
                k:self.convert_numpy(v)
                for k,v in state.items()
                if k!='model_state'
            }
            self.frontend_conn.send(state)
        else:
            print('warning: frontend_conn does not exist')

    def step(self, timestamp):
        """compute one vocoder frame of generation or sampler"""

        state:Dict[str,Any] = {'text':self.text}

        # pause, swap out models, zero text
        if self.needs_model_replace:
            state |= self.do_model_replace()

        # swap out vocoders and possibly change samplerate
        if self.needs_vocoder_replace:
            state |= self.do_vocoder_replace()

        # reset text and model states
        if self.needs_reset and self.model is not None:
            state |= self.do_reset()

        # print(f'{self.frontend_conn=} {threading.get_native_id()=} {os.getpid()=}')

        if self.text_rep is None or self.step_mode==StepMode.PAUSE:
            if len(state):
                self.send_state(state)
            return None
        if self.step_mode==StepMode.GENERATION:
            # with self.profile('step_gen'):
            state |= self.step_gen(timestamp) or {}
        elif self.step_mode==StepMode.SAMPLER:
            with self.profile('step_sampler'):
                state |= self.step_sampler(timestamp) or {}
        # else:
            # print(f'WARNING: {self.step_mode=} in step')

        if self.osc_sender is not None:
            self.osc_sender(state)

        if len(self.latent_biases):
            with torch.inference_mode():
                bias = torch.tensor(self.latent_biases)[None]
                if state.get('latent_t') is not None:
                    l = min(state['latent_t'].shape[1], bias.shape[1])
                    state['latent_t'][:,:l] += bias[:,:l]
                else:
                    print("no latent_t in state")

        # send to frontend
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
                assert self.vocoder is not None, "error: vocoder not loaded"
                # allow creative use of vocoders with mismatched sizes
                rave_dim = self.vocoder.decode_params[0]
                common_dim = min(rave_dim, latent.shape[-1])
                latent_to_rave = torch.zeros(latent.shape[0], rave_dim)
                latent_to_rave[:,:common_dim] = latent[:,:common_dim]
                ### run the vocoder
                with self.profile('vocoder'):
                    audio = self.vocoder.decode(
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

        if self.text is None:
            print('error: text not initialized')
            return None

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
            if self.vocoder is not None:
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


# def main(checkpoint='../rtalign_004_0100.ckpt'):
#     from tungnaa.gui.senders import DirectOSCSender
#     b = Backend(checkpoint, osc_sender=DirectOSCSender())
#     b.start()
#     print('setting text')
#     n = b.set_text('hey, hi, hello, test. text. this is a text.') # returns the token length of the text
#     print(f'text length is {n} tokens')
#     print('setting alignment')
#     i = 0
#     while True:
#         b.set_alignment(torch.randn(1,n).softmax(-1)) # ignored if mode == paint
#         if i==300:
#             print('setting text')
#             n = b.set_text('this is a new text')
#         if i==500:
#             print('setting mode')
#             b.set_mode('paint') # the other available mode is 'infer'

#         while not b.q.empty():
#             r = b.q.get()
#             a = r['align_t'].argmax().item()
#             print('█'*(a+1) + ' '*(n-a-1) + '▏')

#             print(' '.join(f'{x.item():+0.1f}' for x in r['latent_t'][0]))
            
#         time.sleep(1e-2)
#         i += 1

# if __name__=='__main__':
#     fire.Fire(main)
