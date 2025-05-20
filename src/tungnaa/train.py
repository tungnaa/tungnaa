import os
# fix torch device order to be same as nvidia-smi order
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

from pathlib import Path
import random
from collections import defaultdict
import itertools as it
from typing import Tuple

from tqdm import tqdm
import fire

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True

from tungnaa.model import TacotronDecoder
from tungnaa.util import deep_update, get_class_defaults, ConcatSpeakers
from tungnaa.split import get_datasets

class Trainer:
    """Instantiate a Trainer object

    follow with a `train` subcommand to start training

    Args:
        experiment: experiment name
        model_dir: where to store checkpoints
        log_dir: where to store tensorboard logs
        manifest: path to HiFiTTS-style json manifest
        rave_model: path to vocoder .ts file, for logs and determining latent size
        csv: path to additional annotations in a CSV file
            currently based on the format of `jvs_labels_encoder_k7.csv`
            the first column should be the name of the audio file without extension
            the second column will be added to the text, separated by a colon:
            "val:original text"
        concat_speakers: number of utterances to concatenate.
            for each training example, this will load n utterances from the dataset,
            apply any annotations from `csv` or `speaker_annotate`,
            then concatenate the texts and audio.
        strip_quotes: remove all double quotes from text
        speaker_annotate: prepend speaker id to text, as "[speaker]"
        speaker_dataset: when speaker_annotate is True,
            also prepend speaker dataset id to text, as "[dataset:speaker]"
        results_dir: where to store results
        model: dict of model constructor overrides
            e.g. '{text_encoder:{end_tokens:True}, likelihood_type:nsf, flow_blocks:2, nsf_bins:16, prenet_dropout:0.2, rnn_size:1200, decoder_layers:1, frame_channels:11}'
        freeze_text: freeze text encoder
        freeze_embeddings: freeze text encoder embeddings only
        init_text: reinitialize pretrained text encoder
        batch_size: training batch dimensions
        batch_max_tokens: max text length during training, in tokens.
            should be set to usually result in longer text than audio.
            (this depends on the vododer rate and tokens/second of the speech)
        batch_max_frames: max audio length during training, in vocoder frames.
            should be set to usually result in longer text than audio.
            (this depends on the vododer rate and tokens/second of the speech)
        lr: learning rate
        lr_text: override learning rate for text encoder
        adam_betas: AdamW optimizer beta parameters
        adam_eps: AdamW optimizer epsilon parameter
        weight_decay: AdamW optimizer weight decay parameter
        grad_clip: gradient clipping
        seed: random seed
        n_jobs: dataloader parallelism
        device: training device, e.g. 'cpu', 'cuda:0'
        epoch_size: in iterations, None for whole dataset
        valid_size: set size of validation set, in batches
            may be useful if validation set is very large or small
        save_epochs: save checkpoint every so many epochs
        nll_scale: scale NLL loss
        data_portion: fraction of dataset to train on
            (for restricted data experiments)
        debug_loading: debug dataloading
        jit: attempt to compile model with torch.jit.script
        checkpoint: file for resuming training or transfer learning
        resume: if True, resume training, otherwise transfer weights
        rand_text_subs: dict of text substitutions
        style_annotate: probability of prepending style annotations
    """
    def __init__(self, 
        experiment:str, 
        model_dir:Path,
        log_dir:Path, 
        manifest:Path,
        rave_model:Path = None,
        csv:Path=None,
        concat_speakers:int = 0,
        strip_quotes:bool = True,
        speaker_annotate:bool = False,
        speaker_dataset:bool = False,
        results_dir:Path = None,
        model = None, 
        freeze_text:bool = False,
        freeze_embeddings:bool = True,
        init_text:bool = False, 
        batch_size:int = 32,
        # TODO: specify estimated tokens / frame ? 
        batch_max_tokens:int = 256,
        batch_max_frames:int = 512, 
        lr:float = 3e-4,
        lr_text:float = None,
        adam_betas:Tuple[float,float] = (0.9, 0.998),
        adam_eps:float = 1e-08, 
        weight_decay:float = 1e-6,
        grad_clip:float = 5.0,
        seed:int = 0, 
        n_jobs:int = 4,
        device:str = 'cuda:0',
        epoch_size:int = None,
        valid_size:int = None,
        save_epochs:int = 1,
        max_epochs:int = None,
        nll_scale:float = 1, 
        dispersion_scale:float = 0, 
        dispersion_cutoff:float = 0,
        concentration_scale:float = 0, 
        concentration_norm_scale:float = 0, 
        concentration_cutoff:float = 0, 
        # dispersion_l2_scale:float = 0, 
        # concentration_l2_scale:float = 0, 
        # TODO: anneal_prenet = None, # number of epochs to anneal prenet dropout to zero
        data_portion:float = 1,
        debug_loading:bool = False,
        jit:bool = False,
        compile:bool = False,
        checkpoint:Path = None,
        resume:bool = True,
        rand_text_subs = None,
        replace_runs = False,
        style_annotate = 0,
        drop_prefix = None
        ):

        kw = dict(locals()); kw.pop('self')

        kw.pop('checkpoint')
        if checkpoint is not None:
            print(f'loading checkpoint {checkpoint}')
            checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))
            # merges sub dicts, e.g. model hyperparameters
            deep_update(checkpoint['kw'], kw) # provided arguments override stored
            kw = checkpoint['kw']
            load = lambda: self.load_state(checkpoint, resume=resume)
        else:
            load = lambda: None

        # store all hyperparams for checkpointing
        self.kw = kw

        self.best_step = None
        self.best_loss = np.inf

        # get model defaults from model class
        model_cls = TacotronDecoder
        if model is None: model = {}
        assert isinstance(model, dict), """
            model keywords are not a dict. check shell/fire syntax
            """
        kw['model'] = model = get_class_defaults(model_cls) | model

        # assign all arguments to self by default
        self.__dict__.update(kw)
        # mutate some arguments:
        # self.lr_text = self.lr_text or self.lr
        self.model_dir = Path(model_dir) / self.experiment
        self.log_dir = Path(log_dir) / self.experiment
        if results_dir is None:
            self.results_dir = None
        else:
            self.results_dir = Path(results_dir) / self.experiment
        self.manifest = Path(manifest)
        self.device = torch.device(device)

        # filesystem
        for di in (self.model_dir, self.log_dir, self.results_dir):
            if di is not None:
                di.mkdir(parents=True, exist_ok=True)

        if rave_model is None:
            self.rave_model = None
        else:
            self.rave_model = torch.jit.load(rave_model)
            model['block_size'] = self.rave_model.encode_params[3]
            model['frame_channels'] = self.rave_model.latent_size
            
        # random states
        self.seed_random()

        # logging
        self.writer = SummaryWriter(self.log_dir)

        # Trainer state
        self.iteration = 0
        self.exposure = 0
        self.epoch = 0

        if jit:
            kw['model']['script'] = True

        # construct model from arguments 
        self.model = model_cls(**model)
        tqdm.write(repr(self.model))

        tqdm.write(f'{sum(p.numel() for p in self.model.parameters())} parameters')

        if init_text:
            self.model.text_encoder.init()
        if freeze_text:
            self.model.text_encoder.requires_grad_(False)
        if freeze_embeddings:
            for n,m in self.model.text_encoder.named_modules():
                if isinstance(m, torch.nn.Embedding):
                    tqdm.write(f'freezing {n}')
                    m.requires_grad_(False)

        tqdm.write(f'{sum(p.numel() for p in self.model.parameters() if p.requires_grad)} trainable parameters')

        tqdm.write(f'moving model to {self.device}')
        self.model = self.model.to(self.device)

        try:
            self.model.core.attention_rnn.flatten_parameters()
        except Exception:
            tqdm.write('did not flatten attention_rnn parameters')
        try:
            self.model.core.decoder_rnn.flatten_parameters()
            tqdm.write('did not flatten decoder_rnn parameters')
        except Exception:
            pass
      
        # if jit:
            # print(f'scripting model')
            # self.model = torch.jit.script(self.model)
            # self.model.core_loop = torch.jit.script(self.model.core_loop)

        if compile:
            tqdm.write(f'compiling model')
            self.model.core.compile(
                dynamic=True, 
                mode='reduce-overhead'
                )
            # self.model.compile(
            #     dynamic=True, 
            #     mode='reduce-overhead'
            #     )

        tqdm.write(f'constructing datasets')
        self.train_dataset, self.valid_dataset, self.test_dataset = get_datasets(
            manifest, 
            data_portion=data_portion, 
            # concat_speakers=concat_speakers,
            csv_file=csv, 
            max_tokens=batch_max_tokens, 
            max_frames=batch_max_frames,
            speaker_annotate=speaker_annotate,
            speaker_dataset=speaker_dataset,
            strip_quotes=strip_quotes, 
            rave=self.rave_model,
            text_encoder=self.model.text_encoder,
            rand_text_subs=rand_text_subs,
            style_annotate=style_annotate,
            replace_runs=replace_runs
        )
        for tag, ds in (('train', self.train_dataset), ('valid', self.valid_dataset)):
            time_s = 0
            text_c = 0
            for item in ds:
                time_s += item['audio'].shape[1] * self.rave_model.encode_params[3] / self.rave_model.sr
                text_c += len(item['plain_text'])
            tqdm.write(f'{tag} dataset length: {len(ds)} utterances, {time_s} seconds, {text_c} characters')

        self.valid_size = valid_size or len(self.valid_dataset)//batch_size

        if concat_speakers > 1:
            self.train_dataset = ConcatSpeakers(self.train_dataset, concat_speakers, drop_prefix=drop_prefix)
            self.valid_dataset = ConcatSpeakers(self.valid_dataset, concat_speakers, drop_prefix=drop_prefix)
            self.test_dataset = ConcatSpeakers(self.test_dataset, concat_speakers, drop_prefix=drop_prefix)

            
        tqdm.write(f'creating optimizer')
        if self.lr_text is None:
            self.lr_text = self.lr
        params = [{
            'params':p, 'lr':(self.lr_text if 'text_encoder' in n else self.lr)
        } for n, p in self.model.named_parameters()]

        self.opt = torch.optim.AdamW(params,
            self.lr, self.adam_betas, self.adam_eps, self.weight_decay)
        
        load()
        

    @property
    def gpu(self):
        return self.device.type!='cpu'

    def seed_random(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    def set_random_state(self, states):
        # note: GPU rng state not handled
        std_state, np_state, torch_state = states
        random.setstate(std_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)

    @property
    def step(self):
        return self.exposure, self.iteration, self.epoch

    def save(self, fname):
        torch.save(dict(
            kw=self.kw,
            model_state=self.model.state_dict(),
            optimizer_state=self.opt.state_dict(),
            step=self.step,
            best_loss=self.best_loss,
            best_step=self.best_step,
            random_state=(random.getstate(), np.random.get_state(), torch.get_rng_state())
        ), fname)

    def load_state(self, d, resume):
        print(f'{resume=}')
        d = d if hasattr(d, '__getitem__') else torch.load(d)
        sd = d['model_state']
        sd = self.model.update_state_dict(sd)
        if not resume:
            msd = self.model.state_dict()
            for k in list(sd):
                print(k, sd[k].shape, msd[k].shape)
                if k not in msd or sd[k].shape != msd[k].shape:
                    print(f'skipping {k}')
                    sd.pop(k)
        self.model.load_state_dict(sd, strict=resume)
        # self.model.load_state_dict(d['model_state'], strict=resume)
        if resume:
            print('loading optimizer state, RNG state, step counts')
            print("""
            warning: optimizer lr, beta etc are restored with optimizer state,
            even if different values given on the command line, when resume=True
            """)
            self.opt.load_state_dict(d['optimizer_state'])
            self.exposure, self.iteration, self.epoch = d['step']
            self.set_random_state(d['random_state'])
            try:
                self.best_loss = d['best_loss']
                self.best_step = d['best_step']
            except KeyError:
                print('old checkpoint: no best_loss')
        else:
            print('fresh run transferring only model weights')

    def log(self, tag, d):
        # self.writer.add_scalars(tag, d, self.exposure)
        for k,v in d.items():
            self.writer.add_scalar(f'{tag}/{k}', v, self.exposure)
    
    def process_grad(self):
        r = {}
        if self.grad_clip is not None:
            r['grad_l2'] = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip, error_if_nonfinite=True)
        return r

    def get_loss_components(self, result):
        # return {'error': result['mse']}
        return {
            'nll': result['nll'].mean()*self.nll_scale,
            'dispersion': (
                result['dispersion']
                    .clip(self.dispersion_cutoff, 1e9)
                    .mean()
                    *self.dispersion_scale),
            'concentration': (
                result['concentration']
                    .mean()
                    *self.concentration_scale),
            'concentration_norm': (
                result['concentration_norm']
                    .clip(self.concentration_cutoff, 1e9)
                    .mean()
                    *self.concentration_norm_scale),
            # 'dispersion_l2': result['dispersion_l2'].mean()*self.dispersion_l2_scale,
            # 'concentration_l2': result['concentration_l2'].mean()*self.concentration_l2_scale,
            }

    def forward(self, batch):
        audio_lengths = batch['audio_mask'].sum(-1).cpu()
        text_mask = batch['text_mask'].to(self.device, non_blocking=True)
        audio_mask = batch['audio_mask'].to(self.device, non_blocking=True)
        if self.model.text_encoder is not None:
            text = batch['text'].to(self.device, non_blocking=True)
        else:
            text = batch['text_emb']
            if text is None:
                raise ValueError("""
                    no text embeddings in dataset but no text encoder in model
                """)
            text = text.to(self.device, non_blocking=True)
        audio = batch['audio'].to(self.device, non_blocking=True)
        # torch.compiler.cudagraph_mark_step_begin()
        return self.model(
            text, audio, text_mask, audio_mask, audio_lengths, 
            chunk_pad_text=128 if self.compile else None)

    def get_scalars(self, d):
        r = {}
        for k,v in d.items():
            if v.numel()==1:
                r[k] = v.item()
            elif v.shape==(self.batch_size,):
                r[k] = v.mean().item()
        return r

    def _validate(self, valid_loader, ar_mask=None):
        """"""
        pops = defaultdict(list)
        self.model.eval()
        i = 0
        # for batch in tqdm(valid_loader, desc=f'validating epoch {self.epoch}'):
        vs = self.valid_size
        for batch in tqdm(
                it.islice(it.chain.from_iterable(it.repeat(valid_loader)), vs), 
                desc=f'validating epoch {self.epoch}', total=vs):
            with torch.no_grad():
                result = self.forward(batch)
                losses = {
                    k:v.item() 
                    for k,v in self.get_loss_components(result).items()}
                for k,v in losses.items():
                    pops[k].append(v)
                pops['loss'].append(sum(losses.values()))
                # pops['mse'].append(result['mse'].item())
                # pops['nll_fixed'].append(result['nll_fixed'].item())
                # pops['nll_posthoc'].append(result['nll_posthoc'].item())
                for k,v in self.get_scalars(result).items():
                    pops[k].append(v)
                # for k,v in result.items():
                    # NOTE: here (and in train loop), could check for batch size vector and take mean
                    # that would allow processing losses per batch item in training script
                    # if v.numel()==1:
                        # pops[k].append(v.item())
            if i==0:
                self.rich_logs('valid', batch, result)
                i+=1
        return {
            'logs':{k:np.mean(v) for k,v in pops.items()},
            'pops':pops
        }

    def train(self):
        """Entry point to model training"""
        try:
            self._train()
        except Exception as e:
            import traceback; traceback.print_exc()
            import pdb; pdb.post_mortem()

    def _train(self):
        # self.save(self.model_dir / f'{self.epoch:04d}.ckpt')

        # TODO: can remove this?
        def get_collate(ds):
            while True:
                try:
                    return ds.collate_fn()
                except AttributeError:
                    ds = ds.dataset

        train_loader = DataLoader(
            self.train_dataset, self.batch_size,
            shuffle=not isinstance(
                self.train_dataset, torch.utils.data.IterableDataset),
            num_workers=self.n_jobs, pin_memory=self.gpu,
            worker_init_fn=getattr(self.train_dataset, 'worker_init', None),
            persistent_workers=True,
            collate_fn=get_collate(self.train_dataset),
            drop_last=True)

        valid_loader = DataLoader(
            self.valid_dataset, self.batch_size,
            shuffle=False, num_workers=self.n_jobs, pin_memory=self.gpu,
            worker_init_fn=getattr(self.valid_dataset, 'worker_init', None),
            persistent_workers=False,
            collate_fn=get_collate(self.valid_dataset))

        ##### validation loop
        def run_validation():
            if self.debug_loading: return
            logs = self._validate(valid_loader)['logs']
            self.log('valid', logs)
            if logs['loss'] < self.best_loss:
                self.best_loss = logs['loss']
                self.best_step = self.step
                self.save(self.model_dir / f'best.ckpt')

        try:
            epoch_size = self.epoch_size or len(train_loader)
        except TypeError:
            raise ValueError("specify epoch_size when using IterableDataset")

        # validate at initialization
        run_validation()
        
        if self.debug_loading:
            f = open('debug.txt', 'w')

        while self.max_epochs is None or self.epoch < self.max_epochs:
            self.epoch += 1

            ##### training loop
            self.model.train()
            for batch in tqdm(
                # itertools incantation to support epoch_size larger than train set
                # NOTE important to use persistent_workers in DataLoader!
                # otherwise RNG gets repeated
                it.islice(
                    it.chain.from_iterable(it.repeat(train_loader)), epoch_size), 
                desc=f'training epoch {self.epoch}', total=epoch_size
                ):

                self.iteration += 1
                self.exposure += self.batch_size
                logs = {}

                if self.debug_loading:
                    f.write(f'{self.epoch}, {self.iteration}, {list(batch["index"].numpy())}\n')
                    continue

                ### forward+backward+optimizer step ###
                self.opt.zero_grad(set_to_none=True)
                result = self.forward(batch)
                losses = self.get_loss_components(result)
                loss = sum(losses.values())
                loss.backward()
                logs |= self.process_grad()
                self.opt.step()
                ########

                # log loss components
                logs |= {f'loss/{k}':v.item() for k,v in losses.items()}
                # log total loss
                logs |= {'loss':loss.item()}
                # log any other returned scalars
                # logs |= {k:v.item() for k,v in result.items() if v.numel()==1}
                logs |= self.get_scalars(result)
                # other logs
                self.log('train', logs)

            if self.epoch%self.save_epochs == 0: 
                self.save(self.model_dir / f'{self.epoch:04d}.ckpt')
            self.save(self.model_dir / f'last.ckpt')
            run_validation()


    def add_audio(self, tag, audio):
        try:
            sr = self.rave_model.sampling_rate
        except Exception:
            sr = self.rave_model.sr
        self.writer.add_audio(
            tag, audio, 
            global_step=self.epoch, sample_rate=int(sr))

    def wrap_text(self, s, n):
        w = ''
        while s!='':
            w = w+'\n'+s[:n]
            s = s[n:]
        return w

    def rich_logs(self, tag, batch, result):
        """runs RAVE inference and logs audio"""
        z = result['predicted'].detach().cpu()
        gt = result['ground_truth'].detach().cpu()
        a = result['alignment'].detach().cpu()
        am = result['audio_mask'].detach().cpu()
        tm = result['text_mask'].detach().cpu()
        for i in tqdm(range(3), desc='rich logs', leave=False):
            nt = tm[i].sum()
            na = am[i].sum()
            z_i = z[i, :na].T[None]
            gt_i = gt[i, :na].T[None]
            with torch.inference_mode():
                audio_tf = self.rave_model.decode(z_i)[0]
                audio_gt = self.rave_model.decode(gt_i)[0]
            self.add_audio(f'{tag}/audio/tf/{i}', audio_tf)
            self.add_audio(f'{tag}/audio/gt/{i}', audio_gt)
            
            with torch.inference_mode():
                t = result['text'].detach()[i:i+1, :nt]
                z_ar,_ = self.model.inference(
                    t, stop=False, max_steps=int(na*1.2))
                z_ar = z_ar.cpu().transpose(1,2)
                z_ar_zero,_ = self.model.inference(
                    t, stop=False, max_steps=int(na*1.2), temperature=0.0)
                z_ar_zero = z_ar_zero.cpu().transpose(1,2)
                z_ar_half,_ = self.model.inference(
                    t, stop=False, max_steps=int(na*1.2), temperature=0.5)
                z_ar_half = z_ar_half.cpu().transpose(1,2)
                audio_ar = self.rave_model.decode(z_ar)[0]
                audio_ar_zero = self.rave_model.decode(z_ar_zero)[0]
                audio_ar_half = self.rave_model.decode(z_ar_half)[0]
            self.add_audio(f'{tag}/audio/ar/{i}', audio_ar)
            self.add_audio(f'{tag}/audio/ar/zerotemp/{i}', audio_ar_zero)
            self.add_audio(f'{tag}/audio/ar/halftemp/{i}', audio_ar_half)

            fig = plt.figure()
            a_i = a[i, :na, :nt].T
            plt.imshow(
                a_i, 
                interpolation='nearest', 
                aspect='auto', 
                origin='lower')
            plt.title(self.wrap_text(batch['plain_text'][i], 40))
            plt.xlabel(self.wrap_text(batch['audio_path'][i], 64))
            plt.tight_layout()
            self.writer.add_figure(f'{tag}/align/{i}', fig, global_step=self.epoch)

    
def resume(checkpoint, resume:bool=True, **kw):
    """
    Args:
        checkpoint: path to training checkpoint file
        resume: if True, restore optimizer states etc
            otherwise, restore only model weights (for transfer learning)
    """
    d = torch.load(checkpoint, map_location=torch.device('cpu'))
    print(f'loaded checkpoint {checkpoint}')
    # merges sub dicts, e.g. model hyperparameters
    deep_update(d['kw'], kw)
    trainer = Trainer(**d['kw'])
    trainer.load_state(d, resume=resume)

    return trainer

# class Resumable:
#     def __init__(self, checkpoint:Path=None, resume:bool=True, **kw):
#         """
#             checkpoint: path to training checkpoint file
#             resume: if True, restore optimizer states etc
#                 otherwise, restore only model weights (for transfer learning)
#         """
#         if checkpoint is not None:
#             d = torch.load(checkpoint, map_location=torch.device('cpu'))
#             print(f'loaded checkpoint {checkpoint}')
#             # merges sub dicts, e.g. model hyperparameters
#             deep_update(d['kw'], kw)
#             self.trainer = Trainer(**d['kw'])
#             self.trainer.load_state(d, resume=resume)
#         else:
#             self.trainer = Trainer(**kw)

#     def train(self):
#     #     import torch._dynamo.config
#     #     torch._dynamo.config.verbose=True
#     #     self._trainer.compile()
#         self.trainer.train()

#     # def test(self):
#     #     self._trainer.test()

# Resumable.__init__.__doc__ = Resumable.__init__.__doc__ + Trainer.__init__.__doc__
# Resumable.train.__doc__ = Trainer.train.__doc__


if __name__=='__main__':
    # TODO: improve fire-generated help message
    try:   
        fire.Fire()
        # fire.Fire(Resumable)
    except Exception as e:
        import traceback; traceback.print_exc()
        import pdb; pdb.post_mortem()