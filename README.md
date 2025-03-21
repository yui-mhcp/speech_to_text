# :yum: Speech To Text (STT)

Check the [CHANGELOG](https://github.com/yui-mhcp/yui-mhcp/blob/main/CHANGELOG.md) file to have a global overview of the latest modifications! :yum:

## Project structure

```bash
├── architectures            : utilities for model architectures
│   ├── layers               : custom layer implementations
│   ├── transformers         : transformer architecture implementations
│   │   └── whisper_arch.py     : Whisper architecture
│   ├── generation_utils.py  : utilities for text and sequence generation
│   ├── hparams.py           : hyperparameter management
│   └── simple_models.py     : defines classical models such as CNN / RNN / MLP and siamese
├── custom_train_objects     : custom objects used in training / testing
├── loggers                  : logging utilities for tracking experiment progress
├── models                   : main directory for model classes
│   ├── interfaces           : directories for interface classes
│   ├── stt                  : STT implementations
│   │   ├── base_stt.py      : abstract base class for all STT models
│   │   └── whisper.py       : Whisper implementation
│   └── weights_converter.py : utilities to convert weights between different models
├── tests                    : unit and integration tests for model validation
├── utils                    : utility functions for data processing and visualization
├── LICENCE                  : project license file
├── README.md                : this file
├── requirements.txt         : required packages
└── speech_to_text.ipynb     : notebook demonstrating model creation + STT features
```

Check [the main project](https://github.com/yui-mhcp/base_dl_project) for more information about the unextended modules / structure / main classes.

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

### Model weights

The `Whisper` models are automatically downloaded and converted from the `transformers` library.

## Installation and usage

See [the installation guide](https://github.com/yui-mhcp/blob/master/INSTALLATION.md) for a step-by-step installation :smile:

Here is a summary of the installation procedure, if you have a working python environment :
1. Clone this repository: `git clone https://github.com/xxxxx/speech_to_text.git`
2. Go to the root of this repository: `cd speech_to_text`
3. Install requirements: `pip install -r requirements.txt`
4. Open the `speech_to_text` notebook and follow the instructions!

**Important Note** : The `TensorRT-LLM` support for `Whisper` is currently limited to the version `0.15.0` of the library, requiring a `python 3.10` environment. See the installation guide mentionned above for a step-by-step installation ;)

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

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0). See the [LICENSE](LICENSE) file for details.

This license allows you to use, modify, and distribute the code, as long as you include the original copyright and license notice in any copy of the software/source. Additionally, if you modify the code and distribute it, or run it on a server as a service, you must make your modified version available under the same license.

For more information about the AGPL-3.0 license, please visit [the official website](https://www.gnu.org/licenses/agpl-3.0.html)

## Citation

If you find this project useful in your work, please add this citation to give it more visibility! :yum:

```
@misc{yui-mhcp
    author  = {yui},
    title   = {A Deep Learning projects centralization},
    year    = {2021},
    publisher   = {GitHub},
    howpublished    = {\url{https://github.com/yui-mhcp}}
}
```