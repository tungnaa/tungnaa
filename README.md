# Tungnaa Interactive Voice Instruments

Training and GUI inference for interactive artistic text-to-voice models.

We would love to hear your feedback on using Tungnaá in this short survey: [https://forms.gle/F97yhJ1YB5aiZPmn7](https://forms.gle/F97yhJ1YB5aiZPmn7)

# Try it now with [uv](https://docs.astral.sh/uv/) 

` uvx --with "tungnaa[gui]" tungnaa run`

# Installation in a Python environment

`pip install tungnaa[gui]` (if you just want to use Tungnaá)

`pip install tungnaa[train]` (if you want to train your own models)

# Usage

`tungnaa --help`

<!-- # Models from Huggingface -->
<!-- `git clone git@hf.co:intelligent-instruments-Lab/tungnaa` -->

## Running with the Python Audio Engine

- [ ] #todo model selection from the Tungnaa gui, including block size and sample rate that match system (currently not possible to have a SR mismatch)
- [ ] #todo audio device selection from the Tungnaa gui

```bash
tungnaa run --audio-out default
```

use `tungnaa list-devices` to get audio devices by index

## Using SuperCollider, PureData or Max as Audio Engine

If `--latent-audio` switch is enabled, Tungnaa will stream RAVE latent trajectories over a single audio-rate channel, which can be piped into another audio engine running the RAVE vocoder. The piping can be done relatively easily on Linux using JACK, and on MacOS using Blackhole.

SuperCollider:
`sclang supercollider/rtvoice-demo.scd` 

```bash
tungnaa run --latent_audio
```

# Training Models

## vocoder training

using [victor-shepardson RAVE fork](https://github.com/victor-shepardson/RAVE) 

example preprocessing with joining of short files (especially useful for datasets containing many short utterances) 

```bash
rave preprocess \
--input_path /path/to/audio/directory \
--output_path /path/to/tmp/storage/myravedata \
--num_signal 150000 --sampling_rate 48000 \
--join_short_files
```

example transfer learning using [IIL rave-models](https://huggingface.co/Intelligent-Instruments-Lab/rave-models):

```bash
rave train --name 001-my-vocoder-name \
--config rave-models/voice_multi_b2048_r48000/config.gin --config transfer \
--db_path /path/to/tmp/storage/myravedata \
--out_path /path/to/rave/runs \
--transfer_ckpt rave-models/voice_multi_b2048_r48000/version_0/checkpoints/last.ckpt \
--n_signal 150000
--gpu 0
```

example export using sign normalization (latents correlate with louder/brighter sounds):

```bash
rave export --run /path/to/rave/runs/001-my-vocoder-name \
--streaming --normalize_sign --latent_size ...
```

## Tungnaá preprocessing

see `tungnaa prep --help`.

To use datasets other than vctk or hifitts, it may be necessary to add an adapter function in `prep.py`.

example:
```bash
tungnaa prep \
--datasets '{kind:"vctk", path:"/path/to/VCTK"}' \
--rave-path /path/to/rave_streaming.ts \
--out-path /path/to/tmp/dataset_name
```

## training

see `tungnaa trainer --help`

example:
```bash
tungnaa trainer --experiment 001-my-tts-name \
--model-dir /path/for/checkpoints \
--log-dir /path/for/logs \
--manifest /path/to/tmp/dataset_name/manifest.json \
--rave-model /path/to/rave_streaming.ts 
--lr 3e-4 --lr-text 3e-5 --epoch-size 200 --save-epochs 20 \
--device cuda:0 \
train 
```

resume a stopped training: add `--checkpoint /path/to/checkpoint`

transfer learning: add `--checkpoint /path/to/checkpoint --resume False`

### in-text annotations

`--speaker_annotate` prepends the speaker id determined during prepreprocessing. With `--speaker_dataset`, it includes the dataset.

`--csv /path/to/file.csv` accepts a `jvs_labels_encoder_k7.csv`-style CSV file. First column contains the audio filename without extension, second column contains an annotation to be prepended to text.

If you were to use all three options, you would get:

`"csvval:[dataset:speaker] original text"`

# Developing

[uv](https://docs.astral.sh/uv/) is used for packaging and dependency management. After installing `uv`, to get a working dev environment:

1. `git clone git@github.com:tungnaa/tungnaa.git`
2. `cd tungnaa`
3. `uv lock`
4. `uv sync --extra gui --extra train`

To add a dependency, use `uv add`, or edit `pyproject.toml` and then run `uv lock; uv sync`.

Models are stored in a sister [huggingface repo](https://huggingface.co/Intelligent-Instruments-Lab/tungnaa-models-public), or in any similarly structured repo. Tungnaá models should go in `models/tts/mymodel.ckpt`, and be accompanied by a `mymodel.md` file. Vocoders should go in `models/vocoders/myvocoder.ts`.

## docs

run `mkdocs serve` to build and view documentation

run `mkdocs gh-deploy` to deploy to github pages
