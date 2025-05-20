from pathlib import Path
import json
import os
from typing import List, Tuple, Dict, Any
import random

import torch
import torchaudio
import torchaudio.transforms as T
# from transformers import CanineModel
from fire import Fire
from tqdm import tqdm
import pandas as pd

"""
encode all text for one HifiTTS speaker through CANINE (optionally)
and all audio through an exported RAVE model.
Store the result as torch tensors,
and write a new JSON manifest file.
"""

def rate_interp(audio, rate):
    """
    audio: Tensor[channel x time]
    """
    if rate==1:
        return audio
    audio = audio[None] #batch, channel, time
    mode = 'linear' if rate>1 else 'area' 
    audio = torch.nn.functional.interpolate(
        audio, scale_factor=rate, mode=mode)
    return audio[0]

def main(
    datasets:List[Dict[str,Any]]=[], # [{'kind':..., **kwargs}]
    rave_path:str='', # exported RAVE model
    out_path:str='', # output root
    device='cpu',
    aug_n=0, # additional variants
    aug_semitones=0.5, # uniformly distributed +/-
    aug_crop=1024, # samples
    ):

    print(datasets)

    if isinstance(datasets, dict):
        datasets = [datasets]

    out_path = Path(out_path)
    os.makedirs(out_path, exist_ok=True)

    rave_model = torch.jit.load(rave_path)
    try:
        rave_sr = rave_model.sampling_rate
    except Exception:
        rave_sr = rave_model.sr

    # prime RAVE with silence
    rave_model.encode_dist(torch.randn(1,1,rave_sr*2)*1e-5)

    rave_model.to(device)

    resamplers = {}
    def get_resampler(sr):
        if sr==rave_sr:
            return lambda x:x
        if sr not in resamplers:
            resamplers[sr] = T.Resample(
                sr, rave_sr,
                lowpass_filter_width=64,
                rolloff=0.9475937167399596,
                resampling_method="sinc_interp_kaiser",
                beta=14.769656459379492)
        return resamplers[sr]
    
    ####
    #### adapters for specific TTS datasets
    ####
    # first yield the path of a new manifest file to write
    # then for each utterance, yield a dict of:
    #   root dataset path, 
    #   plain text, 
    #   raw audio path relative to dataset root,
    #   dataset:speaker string

    def dadaset(path, manifest='dadaset01-utterances.ods'):
        """dadaset ods format"""
        path = Path(path)

        yield out_path / f'manifest.json' # manifest file to write

        # read csv
        data = pd.read_excel(path / manifest, 'metadata', keep_default_na=False)

        # glob files
        files = path.glob('MIC1/**/*.wav')
        files = {file.stem:file for file in files}

        for row in data.iloc:
            if not row.text: continue
            # print(row)
            file = files[row.pop('filename')]
            audio_path_relative = file.relative_to(path)

            text = row.pop('text')

            # print(row, {**row})
            yield {
                'root_path': path,
                'text': text,
                'audio_path': audio_path_relative,
                'speaker': None,
                **row
            }

    def dadaset_v2(path, manifest='dadaset02-utterances.ods'):
        """dadaset ods format"""
        path = Path(path)

        yield out_path / f'manifest.json' # manifest file to write

        # read csv
        data = pd.read_excel(path / manifest, 'metadata', keep_default_na=False)

        # glob files
        files = path.glob('**/*.wav')
        files = {file.stem:file for file in files}

        for row in data.iloc:
            if not row.text: continue
            # print(row)
            file = files[row.pop('filename')]
            audio_path_relative = file.relative_to(path)

            text = row.pop('text')

            # print(row, {**row})
            yield {
                'root_path': path,
                'text': text,
                'audio_path': audio_path_relative,
                'speaker': None,
                **row
            }

        # raise NotImplementedError

    def vctk(path):
        """all speakers from VCTK"""
        path = Path(path)

        yield out_path / 'vctk.json' # manifest file to write

        for text_path in path.glob('txt/**/*.txt'):
            audio_parent = text_path.relative_to(path).parent
            audio_parent = Path(str(audio_parent).replace(
                'txt/', 'wav48_silence_trimmed/'))
            audio_name = text_path.name.replace('.txt', '_mic1.flac')
            with open(text_path, 'r') as f:
                text = f.read().strip()
            yield {
                'root_path': path,
                'text': text,
                'audio_path': audio_parent / audio_name,
                'speaker': f'vctk:{text_path.parent.name}'
            }

    def hifitts(path, speaker):
        """one specific speaker from HifiTTS"""
        path = Path(path)
        json_path = path / f'{speaker}_manifest_clean_train.json'

        yield out_path / json_path.name # manifest file to write

        with open(json_path) as file:
            for line in file:
                line = json.loads(line)
                yield {
                    'root_path': path,
                    'text': line['text_no_preprocessing'],
                    'audio_path': line['audio_filepath'],
                    'speaker': f'hifitts:{speaker}'
                }

    #### common processing  
    def process(item):
        """item contains:
        text: plain text str
        root_path: path to root dataset
        audio_path: path to audio file as a str, relative to root_path
        speaker: speaker id
        """
        root_path = item.pop('root_path')
        audio_path = item.pop('audio_path')
        orig_audio_path = root_path / audio_path
        audio, sr = torchaudio.load(orig_audio_path)

        audio = get_resampler(sr)(audio)

        # if possible, store the stddev of the RAVE posterior
        save_std = hasattr(rave_model, 'encode_dist')

        # common prefix for each augmented variant
        audio_feat_path = (out_path/audio_path).with_suffix('.pt')
        audio_std_path = audio_feat_path.with_suffix('.params.pt')
        audio_pitch_path = audio_feat_path.with_suffix('.pitch.pt')

        os.makedirs(audio_feat_path.parent, exist_ok=True)
 
        crops = [0]
        rates = [1]
        for _ in range(aug_n):
            crops.append(random.randint(0, aug_crop))
            rates.append(2**((random.random()*2-1)*aug_semitones/12))
        for i,(c,r) in enumerate(zip(crops, rates)):
            save_pitch = process_one(
                i, audio, c, r, 
                audio_feat_path, audio_std_path, audio_pitch_path, 
                save_std)
            
        r = {
            # 'text':item.pop('text'),
            # 'speaker':item.pop('speaker'),
            'audio_path':str(orig_audio_path),
            'audio_feature_path':str(audio_feat_path),
            **item
            }
        
        if save_std:
            r['audio_std_path'] = str(audio_std_path)
        if save_pitch:
            r['audio_pitch_path'] = str(audio_pitch_path)

        return r

    def process_one(
            i, audio, crop, rate, 
            audio_feat_path, audio_std_path, audio_pitch_path, save_std):
        
        audio_feat_path = f'{audio_feat_path}.{i}'
        audio_std_path = f'{audio_std_path}.{i}'
        audio_pitch_path = f'{audio_pitch_path}.{i}'

        audio = audio[:,crop:]
        audio = rate_interp(audio, rate)

        # print(f'{root_path=}, {audio_path=}, {audio_feat_path=}')

        with torch.no_grad():
            audio = audio + torch.randn_like(audio)*1e-5

            if save_std:
                pitch_samp, pitch_params = None, None
                if hasattr(rave_model, 'crop'):
                    # victor-shepardson v1 fork
                    audio_mean, audio_std = rave_model.encode_dist(audio[None])
                    audio_mean = rave_model.crop(audio_mean)
                    audio_std = rave_model.crop(audio_std)
                    audio_samp = (
                        audio_mean + audio_std*torch.randn_like(audio_std))
                    audio_params = audio_mean, audio_std
                elif hasattr(rave_model, 'pitch'):
                    # victor-shepardson v3 fork with pitch
                    d = rave_model.encode_dist(audio[None].to(device))
                    audio_samp = d['z'].cpu()
                    audio_params = (d['z_mean'].cpu(), d['z_std'].cpu())
                    # pitch_samp = d.get('pitch')
                    pitch_params = d.get('pitch_probs')
                    if pitch_params is not None:
                        pitch_params = pitch_params.cpu()
                else: 
                    # victor-shepardson v3 fork
                    audio_samp, audio_params = rave_model.encode_dist(audio[None])
            else:
                # original acids repo
                audio_samp = rave_model.encode(audio[None])
        torch.save(audio_samp, audio_feat_path)
        if save_std:
            torch.save(audio_params, audio_std_path)
        if pitch_params is not None:
            torch.save(pitch_params, audio_pitch_path)
            return True
        return False
            
    for kw in datasets:
        # print(kw)
        adapter = kw.pop('kind')
        # assert adapter in ('vctk', 'hifitts')
        adapter = eval(adapter)
        gen = adapter(**kw)
        out_json_path = next(gen)
        with open(out_json_path, 'w') as outfile:
            for item in tqdm(gen):
                try:
                    result = process(item)
                    json.dump(result, outfile)
                    outfile.write('\n')
                except Exception as e:
                    tqdm.write(f'ERROR: {e} {item}')

if __name__=='__main__':
    Fire(main)