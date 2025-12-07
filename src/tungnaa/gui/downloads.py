import markdown as md
import huggingface_hub as hf
from pathlib import Path

class LocalVocoderInfo():
    def __init__(self, name, path):
        self.name = name
        self.path = path
    def get_path(self):
        return self.path
    
class RemoteVocoderInfo():
    def __init__(self, name, repo, path):
        self.name = name
        self.repo = repo
        self.path = path
    def get_path(self):
        return hf.hf_hub_download(repo_id=self.repo, filename=self.path)

class LocalModelInfo():
    """local model based on .md file and tts/ vocoder/ dir structure"""
    def __init__(self, md_path):
        """"""
        md_path = Path(md_path)
        self.name = md_path.stem
        self._md_path = md_path
        self._md = None

    def get_markdown(self):
        if self._md is None:
            self._md = read_markdown(self._md_path)
        return self._md
    
    def get_tts_path(self):
        return self._md_path.with_suffix('.ckpt')
    
    def get_vocoder_info(self):
        meta = self.get_markdown().Meta
        if meta is None: return None
        name = meta.get("vocoder")
        if name is None: return None
        return LocalVocoderInfo(name, self._md_path.parent.parent/'vocoder'/name)
    
class ManuallyPairedModelInfo():
    """associate TTS and vocoder files given without metadata"""
    def __init__(self, tts_path, vocoder_path):
        self.tts_path = Path(tts_path)
        self.name = self.tts_path.stem
        if vocoder_path is None:
            self.vocoder_info = None
        else:
            vocoder_path = Path(vocoder_path)
            self.vocoder_info = LocalVocoderInfo(vocoder_path.stem, vocoder_path)
    def get_markdown(self):
        return None
    def get_tts_path(self):
        return self.tts_path
    def get_vocoder_info(self):
        return self.vocoder_info
    
class HFModelInfo():
    """model in huggingface repo based on .md file and tts/ vocoder/ dir structure"""
    def __init__(self, repo, md_path):
        """"""
        self.repo = repo
        self._md_path = md_path # remote markdown path
        self.name = Path(md_path).stem # assuming filename is good display name

        self._md = None

    def get_markdown(self):
        """download markdown, parse and return it"""
        if self._md is None:
            print(f"getting {self._md_path} from HF repo {self.repo}")
            path = hf.hf_hub_download(repo_id=self.repo, filename=self._md_path)
            self._md = read_markdown(path)
        return self._md

    def get_tts_path(self):
        """download, cache and return the local path to TTS model"""
        name = Path(self._md_path).stem
        tts_hf = f'models/tts/{name}.ckpt'
        return hf.hf_hub_download(repo_id=self.repo, filename=tts_hf)

    def get_vocoder_info(self):
        """download, cache and return the local path to vocoder if possible"""
        meta = self.get_markdown().Meta
        if meta is None: return None
        name = meta.get("vocoder")
        if name is None: return None
        voc_hf = f'models/vocoder/{name}'
        return RemoteVocoderInfo(
            name, self.repo, voc_hf)


def get_local_model_info(dirs):
    for d in dirs:
        d = Path(d)
        files = d.glob('tts/*.md')
        for f in files:
            yield LocalModelInfo(f)

def get_remote_model_info(repos):
    if not len(repos):
        print('no repos specified')
        return
    fs = hf.HfFileSystem()
    for repo in repos:
        print(f'searching HF {repo=}')
        hf_files = fs.glob(f'{repo}/models/tts/*.md')
        # download and cache each markdown file
        for f in hf_files:
            file = f.split(repo)[-1]
            if file.startswith('/'):
                file = file[1:]
            # print(file)
            yield HFModelInfo(repo, file)   

def read_markdown(f):
    with open(f, 'r') as file:
        text = file.read()
    m = md.Markdown(extensions=['full_yaml_metadata'])
    m.convert(text)
    return m

# def read_remote_models_info(repos):
#     "generate triples of repository, model name, model card markdown"
#     for repo, f in get_remote_metadata_files(repos):
#         k = Path(f).stem
#         yield (repo, k, read_markdown(f))

# def read_local_models_info(model_dirs):
#     "generate triples of models dir, model path, model card markdown"
#     for d, path in get_local_metadata_files(model_dirs):
#         md = read_markdown(path)
#         if md is not None and md.Meta is not None:
#             vocoder_path = md.Meta.get('vocoder')
#         yield (d, path.with_suffix('.ckpt'), vocoder_path, md)

# def dl_model_from_available(available_models, tts, vocoder):
#     for repo,name,markdown in available_models:
#         if name==tts:
#             tts_hf = f'models/tts/{tts}.ckpt'
#             tts = hf.hf_hub_download(repo_id=repo, filename=tts_hf)
#             print(f'{tts=}')
#             if vocoder is None:
#                 voc_hf = f'models/vocoder/{markdown.Meta["vocoder"]}'
#                 vocoder = hf.hf_hub_download(repo_id=repo, filename=voc_hf)
#                 print(f'{vocoder=}')
#             return tts, vocoder, name, markdown
#     raise FileNotFoundError
        
# def dl_model_from_repo(repos, tts, vocoder):
#     return dl_model_from_available(read_remote_models_info(repos), tts, vocoder)

def main(repo='Intelligent-Instruments-Lab/tungnaa-models-public'):
    for item in get_remote_model_info(repo):
        print(item.name)
        print(f'\tTTS: {item.get_tts_path()}')
        print(f'\tvocoder: {item.get_vocoder_path()}')
    # for _,n,m in dl_available_models_info(repo):
    #     print(n)
    #     for k,v in m.Meta.items():
    #         print(f'\t{k}: {v}')
