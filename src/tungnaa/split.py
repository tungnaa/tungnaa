"""
construct and split datasets.
when run as a script, split the audio dataset for vocoder training.
"""

import os

from pathlib import Path
# import random
# from collections import defaultdict
# import itertools as it

# from tqdm import tqdm
import fire

import torch

# TODO: fix this with proper package structure
try:
    from .util import JSONDataset, ConcatSpeakers
except ImportError:
    from util import JSONDataset, ConcatSpeakers

def get_datasets(
        manifest,
        data_portion=1, 
        valid_portion=0.03,
        test_portion=0.02,
        **dataset_kw
        ):
    dataset = JSONDataset(
        manifest, 
        **dataset_kw)
    valid_len = max(8, int(len(dataset)*valid_portion))
    test_len = max(8, int(len(dataset)*test_portion))
    train_len = len(dataset) - valid_len - test_len
    train_dataset, valid_dataset, test_dataset = (
        torch.utils.data.random_split(
            dataset, [train_len, valid_len, test_len], 
            generator=torch.Generator().manual_seed(0)))
    # reduced data size:
    if data_portion != 1:
        train_len = int(train_len*data_portion)
        train_dataset, _ = torch.utils.data.random_split(
            train_dataset,
            [train_len, len(train_dataset) - train_len],
            generator=torch.Generator().manual_seed(0))

    return train_dataset, valid_dataset, test_dataset

def main(
        manifest,
        out_dir,
        split='train',
        concat_speakers=0,
        data_portion=1,
    ):
    if concat_speakers: raise NotImplementedError

    manifest = Path(manifest)
    out_dir = Path(out_dir)

    train_dataset, valid_dataset, test_dataset = get_datasets(
        manifest, csv_file=None, 
        return_path=True,
        data_portion=data_portion, max_tokens=None, max_frames=None)

    prefix = Path(os.path.commonprefix(train_dataset)).parent
    print(prefix)

    if split.startswith('train'):
        dataset = train_dataset
    elif split.startswith('val'):
        dataset = valid_dataset
    elif split.startswith('test'):
        dataset = test_dataset

    #NOTE this needs update when precoputed data augmentation is used

    for i,path in enumerate(dataset):
        path = Path(path)
        dest = out_dir / path.relative_to(prefix).with_suffix(path.suffix)
        print(dest)
        dest.parent.mkdir(exist_ok=True, parents=True)
        dest.symlink_to(path)


if __name__=='__main__':
    try:
        fire.Fire(main)
    except Exception as e:
        import traceback; traceback.print_exc()
        import pdb; pdb.post_mortem()