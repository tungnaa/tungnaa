import markdown as md
import huggingface_hub as hf
from pathlib import Path

def get_local_metadata_files(dirs):
    for d in dirs:
        d = Path(d)
        files = d.glob('tts/*.md')
        for f in files:
            yield d, f 

def get_remote_metadata_files(repos):
    if not len(repos):
        return
    fs = hf.HfFileSystem()
    for repo in repos:
        hf_files = fs.glob(f'{repo}/models/tts/*.md')
        # download and cache each markdown file
        for f in hf_files:
            file = f.split(repo)[-1]
            if file.startswith('/'):
                file = file[1:]
            # print(file)
            yield repo, hf.hf_hub_download(repo_id=repo, filename=file)    

def read_markdown(f):
    with open(f, 'r') as file:
        text = file.read()
    m = md.Markdown(extensions=['full_yaml_metadata'])
    m.convert(text)
    return m

def read_remote_models_info(repos):
    "generate triples of repository, model name, model card markdown"
    for repo, f in get_remote_metadata_files(repos):
        k = Path(f).stem
        yield (repo, k, read_markdown(f))

def read_local_models_info(model_dirs):
    for d, path in get_local_metadata_files(model_dirs):
        k = path.stem
        yield (d, k, read_markdown(path))

def dl_model_from_available(available_models, tts, vocoder):
    for repo,name,markdown in available_models:
        if name==tts:
            tts_hf = f'models/tts/{tts}.ckpt'
            tts = hf.hf_hub_download(repo_id=repo, filename=tts_hf)
            print(f'{tts=}')
            if vocoder is None:
                voc_hf = f'models/vocoder/{markdown.Meta["vocoder"]}'
                vocoder = hf.hf_hub_download(repo_id=repo, filename=voc_hf)
                print(f'{vocoder=}')
            return tts, vocoder, name, markdown
    raise FileNotFoundError
        
def dl_model_from_repo(repos, tts, vocoder):
    return dl_model_from_available(read_remote_models_info(repos), tts, vocoder)

def main(repo='Intelligent-Instruments-Lab/tungnaa-models-public'):
    for _,n,m in dl_available_models_info(repo):
        print(n)
        for k,v in m.Meta.items():
            print(f'\t{k}: {v}')
