# :yum: Speech To Text (STT)

![Python](https://img.shields.io/badge/python-3.12-blue.svg)
![License](https://img.shields.io/badge/licence-AGPL--3.0-green.svg)
![Version](https://img.shields.io/badge/version-1.0.0-orange.svg)

This repository provides an easy-to-use `Speech-To-Text (STT)` API built on top of the `BaseModel` interface, featuring a `keras` implementation of `Whisper`, transcription / word-search utilities and optional `TensorRT-LLM` accelerated inference.

Check the [CHANGELOG](https://github.com/yui-mhcp/yui-mhcp/blob/main/CHANGELOG.md) file to have a global overview of the latest modifications! :yum:

## Project structure

```bash
├── architectures            : utilities for model architectures
│   ├── layers                   : custom layer implementations
│   ├── transformers             : transformer architecture implementations
│   │   ├── *_arch.py                : concrete transformer models (bart, bert, gpt2, mistral, t5, whisper, ...)
│   │   ├── text_transformer_arch.py : base blocks for text-based Transformers
│   │   └── transformer_arch.py      : generic Transformer blocks
│   ├── current_blocks.py        : defines common blocks (e.g., Conv + BN + ReLU)
│   ├── generation_utils.py      : utilities for text and sequence generation
│   ├── hparams.py               : hyperparameter management
│   └── simple_models.py         : defines classical models such as CNN / RNN / MLP and siamese
├── custom_train_objects     : custom objects used in training / testing (callbacks, losses, metrics, optimizers, ...)*
├── docker                   : docker-compose files and Dockerfiles for containerized runs
├── loggers                  : custom utilities for the `logging` module*
├── models                   : main directory for model classes
│   ├── core                     : the `BaseModel` foundation (mixins + saving / migration utils)*
│   ├── stt                      : Speech-To-Text implementations
│   │   ├── base_stt.py              : abstract base class for all STT models (`BaseSTT`)
│   │   └── whisper.py               : Whisper `keras` implementation
│   └── weights_converter.py     : utilities to convert weights between different models
├── pretrained_models        : pretrained / converted checkpoints (e.g. TensorRT-LLM Whisper engines)
├── tests                    : `pytest` suite mirroring the `utils` / `loggers` and model tree*
├── utils                    : shared data-processing / keras utilities*
├── audio_en.wav                : example audio file used by the notebook
├── convert_checkpoint-0.18.py  : HuggingFace -> TensorRT-LLM checkpoint conversion (TRT-LLM 0.18 API)
├── convert_checkpoint-0.19.py  : HuggingFace -> TensorRT-LLM checkpoint conversion (TRT-LLM 0.19 API)
├── speech_to_text.ipynb        : notebook demonstrating model creation + STT features
├── CHANGELOG.md                : project-specific changelog
├── CITATION.cff                : citation metadata
├── CONTRIBUTING.md             : contribution guidelines
├── INSTALLATION.md             : shared GPU environment installation guide
├── LICENCE                     : project license file
├── pyproject.toml              : project metadata, dependencies and tooling config
└── README.md                   : this file
```

\* The `loggers`, `utils`, `custom_train_objects`, `models/core` and most of the `tests` tree are the shared foundation maintained in the [base_dl_project](https://github.com/yui-mhcp/base_dl_project) and [data_processing](https://github.com/yui-mhcp/data_processing) repositories — check them for more information on these modules / main classes.

## Available features

- **Speech-To-Text** (module `models.stt`) :

| Feature   | Function / class   | Description |
| :-------- | :---------------- | :---------- |
| Speech-To-Text    | `stt`             | Perform `STT` on audio / video files      |
| Search            | `search`          | Search for words in audio / video and display timestamps |

The `speech_to_text` notebook provides a concrete demonstration of the `stt` and `search` functions.

## Available models

### Model architectures

Available architectures:
- [Whisper](https://github.com/openai/whisper): OpenAI's Whisper multilingual STT model with transformer architecture

## Installation and usage

See [the installation guide](INSTALLATION.md) for a step-by-step setup of the shared GPU environment (NVIDIA driver, CUDA, `mamba` and the deep-learning backends) :smile:

Here is a summary of the installation procedure, once your python environment is ready :
1. Clone this repository : `git clone https://github.com/yui-mhcp/speech_to_text.git`
2. Go to the root of this repository : `cd speech_to_text`
3. Install the package with a backend : `pip install -e .[tf]` (or `pip install -e .[torch]`)
4. Open the `speech_to_text` notebook and follow the instructions !

Like the [base_dl_project](https://github.com/yui-mhcp/base_dl_project), this project trains / runs real models, so it needs a **keras backend**. The audio, text and `transformers` dependencies (required by every STT model) are installed by default — you only pick the backend extra matching your setup :

```bash
pip install -e .[tf]         # keras + tensorflow (+ torch for checkpoint conversion)
pip install -e .[torch]      # keras + torch
pip install -e .[keras]      # alias of [tf]

# optional helpers :
pip install -e .[image]      # cv2 / pillow-based utilities (utils/image)
pip install -e .[datasets]   # utils/datasets (tensorflow-datasets, pandas)
pip install -e .[dev]        # test tooling (pytest & plugins)

# extras can be combined :
pip install -e .[tf,image,datasets]
```

The backend is selected at runtime through the `KERAS_BACKEND` environment variable (e.g., `os.environ['KERAS_BACKEND'] = 'tensorflow'`), as shown at the top of the notebook.

**Important Note** : converting the official `Whisper` checkpoints (pytorch → keras) requires `torch`, which is why it is pulled by the `[tf]` extra as well. For **accelerated inference**, the [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) support for `Whisper` targets `tensorrt_llm < 1.3` on a `python 3.12` environment ; it is installed separately from the NVIDIA package index — see the [installation guide](INSTALLATION.md) (which also notes the `1.2.1` int8 caveat on Blackwell) and the `convert_checkpoint-0.*.py` helper scripts. ;)

### Quickstart

```python
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

from models.stt import Whisper

# 1. Build a pretrained Whisper model (downloaded / converted from `transformers`)
model = Whisper(pretrained = 'openai/whisper-base', lang = 'multi', nom = 'whisper-base')
print(model)

# 2. Transcribe an audio / video file
result = model.predict('audio_en.wav')
```

See the `speech_to_text.ipynb` notebook for the full walkthrough (model building and transcription). :smile:

### Testing

The tests use [`pytest`](https://docs.pytest.org/) and live in the `tests/` directory, mirroring the `utils` / `loggers` and model tree.

```bash
pip install -e .[dev]        # pytest, pytest-cov, pytest-xdist, pytest-timeout

pytest                       # run the whole suite
pytest -n auto               # run in parallel (pytest-xdist)
pytest -m "not slow"         # skip the slow / heavy tests
pytest --cov                 # run with coverage report
pytest tests/models          # run a single subpackage
```

Tests are annotated with markers (declared in `pyproject.toml`) so the suite adapts to your environment : the `tensorflow`, `torch`, `keras`, `cv2` and `gpu` markers are **auto-skipped** when the corresponding dependency (or hardware) is missing, meaning you can run the tests for the backend you installed without pulling every dependency.

## TO-DO list:

- [x] Make the TO-DO list
- [x] Comment the code
- [x] Add multilingual model support (`Whisper`)
- [x] Add Beam-Search text decoding
- [ ] Add streaming support
- [x] Convert `Whisper` pretrained models from the `transformers` hub
- [x] Support [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) for inference

## Search and partial alignment

Even though `Whisper` produces high-quality transcriptions, it can still make mistakes, making exact-match searches ineffective. To address this limitation, the proposed `search` method leverages the `Edit` distance to compute a similarity score between the search text and the produced transcription. This allows matches to be defined based on a tolerance threshold rather than exact matching!

For instance, searching *cat* in *the ct is on the chair* will not find an exact match for *cat*, while *ct* has only 1 mismatch (the missing *a*).

The Levenshtein distance produces an alignment between *cat* and *ct* with a distance matrix:

|   |   | c | t |
|:-:|:-:|:-:|:-:|
|   | 0 | 1 | 2 |
| c | 1 | 0 | 1 |
| a | 2 | 1 | 1 |
| t | 3 | 2 | 1 |

The bottom-right value is 1, which represents the total number of operations (addition/deletion/replacements) needed to transform the hypothesis (`ct`) into the reference (`cat`).

The value at index **i, j** is the minimum between:
- matrix[i-1][j] + deletion cost of character i (in hypothesis)
- matrix[i-1][j-1] + replacement cost of character i (in hypothesis) and character j (of truth) (equal to 0 if both are the same character)
- matrix[i][j-1] + insertion cost of character j (of hypothesis)

Note: To simplify the examples, all costs have been set to 1, but they can be specified in the `edit_distance` function (e.g., punctuation may have a cost of 0).

The objective is to align *cat* at all positions of the transcript (*the ct is*). For this purpose, the solution sets the 1st line to 0, allowing alignment at each position without penalizing the position of the alignment:

|   |   | t | h | e |   | c | t |   | i | s |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|   | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| c | 1 | 1 | 1 | 1 | 1 | 0 | 1 | 1 | 1 | 1 |
| a | 2 | 2 | 2 | 2 | 2 | 1 | 1 | 2 | 2 | 2 |
| t | 3 | 2 | 3 | 3 | 3 | 2 | 1 | 2 | 3 | 3 |

Note: Scores will be more relevant for longer search terms, as they're less influenced by small variations.

An example is provided in the `speech_to_text` notebook to better illustrate how the search works.

## Notes and references

This section proposes useful projects, papers and tutorials to learn more about `Speech-To-Text (STT)` techniques, models and frameworks.

### Key Concepts in STT

1. **Acoustic Modeling**: Converting audio signals into phonetic representations
2. **Language Modeling**: Determining the probability of word sequences
3. **Feature Extraction**: Converting raw audio into spectrograms or MFCCs (Mel-frequency cepstral coefficients)
4. **Decoding**: Translating acoustic features into text transcriptions

### Popular STT Approaches

1. **CTC (Connectionist Temporal Classification)**: Used in DeepSpeech and Jasper
2. **Seq2Seq with Attention**: Used in models like Listen, Attend and Spell
3. **Transformer-based approaches**: Used in Whisper and SpeechT5
4. **RNN-Transducer (RNN-T)**: Used in production systems like Google's speech recognition

### Papers

- [Self-Supervised Learning for Speech Recognition](https://arxiv.org/abs/2006.11477): Overview paper on self-supervised approaches
- [Deep Speech 2: End-to-End Speech Recognition in English and Mandarin](https://www.paperswithcode.com/paper/deep-speech-2-end-to-end-speech-recognition): Original DeepSpeech2 paper
- [Jasper: An End-to-End Convolutional Neural Acoustic Model](https://www.paperswithcode.com/paper/jasper-an-end-to-end-convolutional-neural): The original Jasper paper
- [Speech-Transformer: A No-Recurrence Sequence-to-Sequence Model for Speech Recognition](https://ieeexplore.ieee.org/document/8462506): Original SpeechTransformer paper
- [A technique for computer detection and correction of spelling errors](https://dl.acm.org/doi/10.1145/363958.363994): Levenshtein distance paper
- [RNN-T for Latency Controlled ASR WITH IMPROVED BEAM SEARCH](https://www.arxiv-vanity.com/papers/1911.01629/): RNN-Transducer paper
- [Conformer: Convolution-augmented Transformer for Speech Recognition](https://arxiv.org/abs/2005.08100): Conformer original paper
- [Whisper: Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356): OpenAI's Whisper paper

### Tutorials

- [Speech Recognition with TensorFlow](https://www.tensorflow.org/tutorials/audio/simple_audio): Official TensorFlow tutorial to get started with audio processing
- [Introduction to Automatic Speech Recognition](https://huggingface.co/learn/audio-course/chapter1/introduction): Hugging Face course on ASR basics
- [Speech Recognition with Wav2Vec2](https://huggingface.co/blog/fine-tune-wav2vec2-english): Fine-tuning Wav2Vec2 for English ASR
- [End-to-End Speech Recognition Systems](https://distill.pub/2017/ctc/): Visual explanation of CTC and end-to-end systems
- [Keras tutorial](https://keras.io/examples/audio/transformer_asr/): Tutorial on speech recognition with Transformers
- [Levenshtein distance computation](https://blog.paperspace.com/measuring-text-similarity-using-levenshtein-distance/): A Step-by-Step computation of the Levenshtein distance
- [NVIDIA NeMo project](https://developer.nvidia.com/nvidia-nemo): Main website for NVIDIA NeMo project, containing many tutorials on NLP (ASR, TTS, etc.)

### GitHub projects

- [LibriSpeech ASR with PyTorch](https://github.com/pytorch/audio/tree/main/examples/asr): PyTorch example using the LibriSpeech dataset
- [Mozilla DeepSpeech Examples](https://github.com/mozilla/DeepSpeech/tree/master/examples): Practical examples using Mozilla's implementation
- [Whisper Fine-Tuning Examples](https://github.com/huggingface/transformers/tree/main/examples/pytorch/speech-recognition): Hugging Face examples for fine-tuning Whisper
- [NVIDIA's Jasper project](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechRecognition/Jasper): Original Jasper code
- [NVIDIA's NeMo project](https://github.com/NVIDIA/NeMo): Provides a PyTorch implementation of the `Conformer` and `RNN-T` models
- [Automatic Speech Recognition project](https://github.com/rolczynski/Automatic-Speech-Recognition): DeepSpeech2 implementation
- [OpenAI's Whisper](https://github.com/openai/whisper): The official OpenAI implementation of Whisper (in PyTorch)
- [ESPnet](https://github.com/espnet/espnet): End-to-End Speech Processing Toolkit with various ASR implementations
- [SpeechBrain](https://github.com/speechbrain/speechbrain): PyTorch-based speech toolkit covering various speech tasks

## Contacts and licence

Contacts:
- **Mail**: `yui-mhcp@tutanota.com`
- **[Discord](https://discord.com)**: yui0732

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0). See the [LICENCE](LICENCE) file for details.

This license allows you to use, modify, and distribute the code, as long as you include the original copyright and license notice in any copy of the software/source. Additionally, if you modify the code and distribute it, or run it on a server as a service, you must make your modified version available under the same license.

For more information about the AGPL-3.0 license, please visit [the official website](https://www.gnu.org/licenses/agpl-3.0.html)

## Citation

If you find this project useful in your work, please add this citation to give it more visibility! :yum:

```
@misc{yui-mhcp,
    author  = {yui},
    title   = {A Deep Learning projects centralization},
    year    = {2021},
    publisher   = {GitHub},
    howpublished    = {\url{https://github.com/yui-mhcp}}
}
```
