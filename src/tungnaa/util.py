from collections.abc import Mapping
import inspect
import json
import glob
import re
from pathlib import Path
import random
from collections import defaultdict

import torch

def get_char_runs(s):
    c_last = None
    runs = []
    for c in s:
        if c == c_last:
            runs[-1] += c
        else:
            runs.append(c)
        c_last = c
    return runs

def pad1(t, n):
    if t.shape[1] >= n:
        t = t[:, :n]
    else:
        z = t.new_zeros((t.shape[0], n-t.shape[1], *t.shape[2:]))
        t = torch.cat((t, z), 1)
    return t

_quotes_re = re.compile(r'"|”|“')

def choose_path(path, i=None):
    # choose one path with given prefix
    paths = glob.glob(f'{path}*')
    # for p in paths: print(p) ## DEBUG
    if i is None:
        i = random.randint(0, len(paths)-1)
    return paths[i]

class JSONDataset(torch.utils.data.Dataset):
    """
    """
    def __init__(self, 
            manifest_file, csv_file, max_tokens, max_frames, 
            speaker_annotate=False, speaker_dataset=False, strip_quotes=False,
            rave=None, text_encoder=None, return_path=False, rand_text_subs=None, style_annotate=0, replace_runs=False):
        super().__init__()
        """
        manifest_file: hifi-tts style json manifest file (see prep.py)
        csv_file: csv file where first column is audio filename, 
            other columns contain additional metadata

        rand_text_subs: list of (originals, replacement) token pairs
        replace_runs: if True, always replace runs of the same token together
        """
        self.root_dir = Path(manifest_file).parent
        with open(manifest_file) as f:
            self.index = [json.loads(line) for line in f]

        self.csv_data = {}
        if csv_file is not None:
            with open(csv_file) as f:
                for line in f:
                    k,*vs = line.strip().split(',')
                    self.csv_data[k] = vs
            # print(self.csv_data)###DEBUG
        self.max_frames = max_frames
        self.max_tokens = max_tokens
        self.speaker_annotate = speaker_annotate
        self.speaker_dataset = speaker_dataset
        self.style_annotate = style_annotate
        self.strip_quotes = strip_quotes
        self.rave = rave # needed for pitch models
        self.text_encoder = text_encoder
        self.return_path = return_path

        if rand_text_subs is not None:
            subs = defaultdict(list)
            for ogs,t in rand_text_subs:
                for c in ogs:
                    subs[c].append(t)
            self.rand_text_subs = dict(subs)
        else:
            self.rand_text_subs = None
        self.replace_runs = replace_runs

    def worker_init(self, i):
        s = torch.initial_seed()
        random.seed(s)

    def __getitem__(self, i):
        item = self.index[i]
        if self.return_path:
            return item['audio_path']
        
        # pre-computed data augmentation
        # if not exists, choose random
        # TODO

        if 'audio_std_path' in item:
            # data augmentation if RAVE prep included posterior stddevs
            audio, stddev = torch.load(choose_path(
                self.root_dir / item['audio_std_path']))
            stddev.mul_(torch.randn_like(stddev))
            audio.add_(stddev)
        else:
            print(f'{item=}')
            audio = torch.load(choose_path(
                self.root_dir / item['audio_feature_path']))

        if 'audio_pitch_path' in item:
            pitch_probs = torch.load(choose_path(
                self.root_dir / item['audio_pitch_path']))
            pitch_bins = torch.distributions.Categorical(pitch_probs).sample()
            pitch_hz = self.rave.pitch_encoder.bins_to_frequency(pitch_bins)
            # pitch becomes first latent.
            # print(pitch_hz.shape, audio.shape)
            # slice off the last frame of audio if needed
            # TODO: look into this discrepancy when input sizes aren't round...
            # print(pitch_hz.shape, audio.shape)
            audio = torch.cat((pitch_hz, audio[...,:pitch_hz.shape[-1]]), 1)

        text = item['text']

        if self.strip_quotes:
            text = re.sub(_quotes_re, '', text)

        if self.rand_text_subs is not None:
            rate = random.random()**2
            if self.replace_runs:
                runs = []
                for run in get_char_runs(text):
                    c, n = run[0], len(run)
                    if c in self.rand_text_subs and random.random() < rate:
                        toks = self.rand_text_subs[c]
                        runs.append(random.choice(toks) * n)
                    else:
                        runs.append(run)
                text = ''.join(runs)
            else:
                text = list(text)
                for i in range(len(text)):
                    if text[i] in self.rand_text_subs and random.random() < rate:
                        toks = self.rand_text_subs[text[i]]
                        text[i] = random.choice(toks)
                text = ''.join(text)
            # print(text)

        if self.speaker_annotate:
            speaker = item['speaker']
            if not self.speaker_dataset:
                speaker = speaker.split(':')[1]
            text = f'[{speaker}] {text}'

        if self.style_annotate:
            if random.random() < self.style_annotate:
                style = item['style']
                # capitalize first letter of style
                if style=='neutral':
                    style = '|'
                elif style=='aggressive':
                    style = '+'
                elif style=='happy':
                    style = '^'
                elif style=='worried':
                    style = '&'
                else:
                    print(f'warning: unrecognized style "{style}"')
                    style = ''
            else:
                style = '%'
            text = style + text
            

        if len(self.csv_data):
            k = item['audio_path']
            # k = Path(k).name.replace('.flac', '.wav')
            k = Path(k).name.split('.')[0]
            # assert k in self.csv_data, k
            if k not in self.csv_data:
                print(f'WARNING: {k} not found in csv')
            else:
                text = f'{self.csv_data[k][0]}:{text}'
            # print(text) ###DEBUG
 
        r = {
            'index': i,
            'plain_text': text,
            # batch x time x channel
            'audio': audio.transpose(1,2),
            'audio_path': item['audio_path'],
        }
        if 'text_feature_path' in item:
            if self.speaker_annotate:
                raise NotImplementedError
            # batch x time x channel
            r['emb_text'] = torch.load(
                self.root_dir / item['text_feature_path'])
        return r
    
    def __len__(self):
        return len(self.index)

    def collate_fn(self):
        """
        returns a function suitable for the collate_fn argument of a torch DataLoader
        """
        def collate(batch):
            """
            Args:
                batch: list of dicts which are single data points
                
            Return:
                dict of batch tensors
            """
            n_text = min(self.max_tokens, max(len(b['plain_text']) for b in batch))
            n_audio = min(self.max_frames, max(b['audio'].shape[1] for b in batch))
            text_idxs = []
            text_embs = []
            audio_ts = []
            text_masks = []
            audio_masks = []
            # index_ts = []
            for b in batch:
                text_idx, *_ = self.text_encoder.tokenize(b['plain_text'])
                # text_idx = torch.tensor(
                    # [[ord(char) for char in b['plain_text']]])
                if 'emb_text' in b:
                    text_emb = b['emb_text']
                    assert text_idx.shape[1] == text_emb.shape[1]
                    text_embs.append(pad1(text_emb, n_text))
                text_masks.append(torch.arange(n_text) < text_idx.shape[1])
                text_idxs.append(pad1(text_idx, n_text))
                audio = b['audio']
                audio_masks.append(torch.arange(n_audio) < audio.shape[1])
                audio_ts.append(pad1(audio, n_audio))
                # index_ts.append(torch.LongTensor((b['index'],)))
            return {
                # 'index': torch.cat(index_ts, 0),
                'plain_text': [b['plain_text'] for b in batch],
                'audio_path': [b['audio_path'] for b in batch],
                'text': torch.cat(text_idxs, 0),
                'text_emb': torch.cat(text_embs, 0) if len(text_embs) else None,
                'audio': torch.cat(audio_ts, 0),
                'text_mask': torch.stack(text_masks, 0),
                'audio_mask': torch.stack(audio_masks, 0),
            }

        return collate

class ConcatSpeakers(torch.utils.data.IterableDataset):
    """Iterable-style dataset which combines two JSONDatasets"""
    def __init__(self, dataset, n=2, end_tokens=True, drop_prefix=None):
        super().__init__()
        """
        """
        self.dataset = dataset
        self.n = n
        self.end_tokens = end_tokens
        self.drop_prefix = drop_prefix #(prob, prefix)
        self.iter_count = 0
  
    def worker_init(self, i):
        s = torch.initial_seed() ^ self.iter_count
        random.seed(s)
        torch.random.manual_seed(s)

    def __iter__(self):
        self.iter_count += 1
        return self
    def __next__(self):
        # get a random element from each dataset
        items = [
            self.dataset[torch.randint(len(self.dataset),size=(1,)).item()]
            for _ in range(self.n)
        ]
        # concat audio, text
        # print(items)

        text = ''.join(item['plain_text'] for item in items)

        # randomly drop certain prefixes -- use this for dropping 'wildcard' annotations from the first utterance only, while always including them for subsequent utterances
        if self.drop_prefix is not None:
            s, p = self.drop_prefix
            if text.startswith(s) and random.random() < p:
                text = text[len(s):]

        return {
            'audio': torch.cat([item['audio'] for item in items], 1),
            'plain_text': text,
            'audio_path': ';'.join(item['audio_path'] for item in items)
        }
    
    # @property
    # def dataset(self):
    #     # compat with split
    #     return self
  
    def collate_fn(self):
        return self.datasets[0].dataset.collate_fn()
# class ConcatSpeakers(torch.utils.data.IterableDataset):
#     """Iterable-style dataset which combines two JSONDatasets"""
#     def __init__(self, manifest_files, split, max_tokens, max_frames,
#             valid_portion=0.03, test_portion=0.02, csv_file=None):
#         super().__init__()
#         """
#         """
#         self.datasets = []
#         for manifest in manifest_files:
#             dataset = JSONDataset(
#                 manifest, csv_file, max_tokens, max_frames, speaker_annotate=True) 
#             valid_len = max(8, int(len(dataset)*valid_portion))
#             test_len = max(8, int(len(dataset)*test_portion))
#             train_len = len(dataset) - valid_len - test_len
#             splits = torch.utils.data.random_split(
#                 dataset, [train_len, valid_len, test_len], 
#                 generator=torch.Generator().manual_seed(0))
#             i = {'train':0, 'valid':1, 'test':2}[split]
#             self.datasets.append(splits[i])
  
#     def __iter__(self):
#         return self
#     def __next__(self):
#         # get a random element from each dataset
#         items = [
#             ds[torch.randint(len(ds),size=(1,)).item()]
#             for ds in self.datasets
#         ]
#         # concat audio, text
#         # print(items)

#         # TODO: remove internal start / end tokens if using

#         return {
#             'audio': torch.cat([item['audio'] for item in items], 1),
#             'plain_text': ''.join(item['plain_text'] for item in items)
#         }
    
#     @property
#     def dataset(self):
#         # compat with split
#         return self
  
#     def collate_fn(self):
#         return self.datasets[0].dataset.collate_fn()


def get_function_defaults(fn):
    """get dict of name:default for a function's arguments"""
    s = inspect.signature(fn)
    return {k:v.default for k,v in s.parameters.items()}

def get_class_defaults(cls):
    """get the default argument values of a class constructor"""
    d = get_function_defaults(getattr(cls, '__init__'))
    # ignore `self` argument, insist on default values
    try:
        d.pop('self')
    except KeyError:
        raise ValueError("""
            no `self` argument found in class __init__
        """)
    assert [v is not inspect._empty for v in d.values()], """
            get_class_defaults should be used on constructors with keyword arguments only.
        """
    return d

def deep_update(a, b):
    """
    in-place update a with contents of b, recursively for nested Mapping objects.
    """
    for k in b:
        if k in a and isinstance(a[k], Mapping) and isinstance(b[k], Mapping):
            deep_update(a[k], b[k])
        else:
            a[k] = b[k]


# def arg_to_set(x):
#     """convert None to empty set, iterable to set, or scalar to set with one item"""
#     if x is None:
#         return set()
#     elif not hasattr(x, '__iter__'):
#         return {x}
#     else:
#         return set(x)