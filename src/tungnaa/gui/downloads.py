import markdown as md
import huggingface_hub as hf
from pathlib import Path

def get_metadata_files(repo):
    fs = hf.HfFileSystem()
    hf_files = fs.glob(f'{repo}/models/tts/*.md')
    # download and cache each markdown file
    for f in hf_files:
        file = f.split(repo)[-1]
        if file.startswith('/'):
            file = file[1:]
        # print(file)
        yield hf.hf_hub_download(repo_id=repo, filename=file)    

def read_markdown(f):
    with open(f, 'r') as file:
        text = file.read()
    m = md.Markdown(extensions=['full_yaml_metadata'])
    m.convert(text)
    return m

def get_markdown(repo):
    for f in get_metadata_files(repo):
        k = Path(f).stem
        yield (k, read_markdown(f))

def dl_model(repo, tts, vocoder):
    for n,m in get_markdown(repo):
        if n==tts:
            tts_hf = f'models/tts/{tts}.ckpt'
            tts = hf.hf_hub_download(repo_id=repo, filename=tts_hf)
            print(f'{tts=}')
            if vocoder is None:
                voc_hf = f'models/vocoder/{m.Meta["vocoder"]}'
                vocoder = hf.hf_hub_download(repo_id=repo, filename=voc_hf)
                print(f'{vocoder=}')
            return tts, vocoder, n, m

def main(repo='Intelligent-Instruments-Lab/tungnaa-models-public'):
    for n,m in get_markdown(repo):
        print(n)
        for k,v in m.Meta.items():
            print(f'\t{k}: {v}')
