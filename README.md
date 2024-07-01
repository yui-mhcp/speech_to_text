# :yum: Speech To Text (STT)

Check the [CHANGELOG](https://github.com/yui-mhcp/yui-mhcp/blob/main/CHANGELOG.md) file to have a global overview of the latest modifications ! :yum:

## Project structure

```bash
.
├── custom_architectures
│   ├── transformers_arch
│   │   └── whisper_arch.py     : Whisper architecture
│   ├── deep_speech_2_arch.py   : DeepSpeech2 architecture
│   ├── generation_utils.py : inference methods used for Whisper
│   └── jasper_arch.py      : Jasper architecture
├── custom_layers
├── custom_train_objects
├── loggers
├── models
│   ├── stt
│   │   ├── base_stt.py             : abstract class for STT models
│   │   ├── deep_speech.py          : Deep Speech 2 main class
│   │   ├── jasper.py               : Jasper main class
│   │   └── whisper.py              : Whisper main class
├── pretrained_models
├── unitests
├── utils
└── speech_to_text.ipynb
```

Check [the main project](https://github.com/yui-mhcp/base_dl_project) for more information about the unextended modules / structure / main classes. 

## Available features

- **Speech-To-Text** (module `models.stt`) :

| Feature   | Fuction / class   | Description |
| :-------- | :---------------- | :---------- |
| Speech-To-Text    | `stt`             | Perform `STT` on audio / video      |
| search            | `search`          | Search a word on audio / video and display timestamps |

The `speech_to_text` notebook provides a concrete demonstration of the `stt` and `search` functions

## Available models

### Model architectures

Available architectures : 
- `CTC decoders` :
    - [DeepSpeech2](https://www.paperswithcode.com/paper/deep-speech-2-end-to-end-speech-recognition)
    - [Jasper](https://www.paperswithcode.com/paper/jasper-an-end-to-end-convolutional-neural)
- `Generative models` :
    - [Whisper](https://github.com/openai/whisper) : OpenAI's Whisper multilingual STT model.

### Model weights

The `Whisper` models are automatically downloaded and converted into the `keras` implementation.

## Installation and usage

1. Clone this repository : `git clone https://github.com/xxxxx/speech_to_text.git`
2. Go to the root of this repository : `cd speech_to_text`
3. Install requirements : `pip install -r requirements.txt`
4. Open `speech_to_text` notebook and follow the instruction !

## TO-DO list :

- [x] Make the TO-DO list
- [x] Comment the code
- [x] Add support for pretrained `DeepSpeech2`
- [x] Add support for pretrained `Jasper` (from [NVIDIA's official repository](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechRecognition/Jasper))
- [x] Add multilingual model support (`Whisper`)
- [x] Add Beam-Search text decoding
- [ ] Add streaming support 
- [x] Convert `Whisper` pretrained models from the `transformers` hub (currently, only the official `Whisper` weights are supported)


## Search and partial alignment

Even though `Whisper` produces really good quality transcription, it can still make some mistakes in its transcript, making exact-match search not working. To solve this limitation, the proposed `search` method leverages the `Edit` distance to compute a score between the search text, and the produced transcription. This allows to define matches based on a tolerance threshold, rather than exact matching !

For instance, searching *cat* in *the ct is on the chair* will not find *cat*, while *ct* has only 1 mismatch (the missing *a*). 

The Levenshtein distance produces an alignment between *cat* and *ct* with a distance matrix (I put a link for step-by-step computing of this matrix) : 

|   |   | c | t |
|:-:|:-:|:-:|:-:|
|   | 0 | 1 | 2 |
| c | 1 | 0 | 1 |
| a | 2 | 1 | 1 |
| t | 3 | 2 | 1 |

The bottom-right value is 1, which represents the total number of operations (addition / deletion / replacements) to obtain the reference (`cat`) from the hypothesis (`ct`).

The value at index **i, j** is the minimum between :
- matrix[i-1][j]    + deletion cost of caracter i (in hypothesis)
- matrix[i-1][j-1]  + replacement cost of caracter i (in hypothesis) and caracter j (of truth) (equal to 0 if both are same caracter)
- matrix[i][j-1]    + insertion cost of caracter j (of hypothesis)

Note : to simplify the examples, all costs have been set to 1, but they can be specified in the `edit_distance` function (e.g., punctuation may have a cost of 0). 

The objective is now to align *cat* at all position of `truth` (*the ct is*). For this purpose, the proposed solution is to set the 1st line to 0, such that it will try to align *cat* at each position without penalizing the position of the alignment. 

|   |   | t | h | e |   | c | t |   | i | s |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|   | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| c | 1 | 1 | 1 | 1 | 1 | 0 | 1 | 1 | 1 | 1 |
| a | 2 | 2 | 2 | 2 | 2 | 1 | 1 | 2 | 2 | 2 |
| t | 3 | 2 | 3 | 3 | 3 | 2 | 1 | 2 | 3 | 3 |

Note : scores will be less influenced (so more relevant) if the searched text is longer. 

An example is provided in the `speech_to_text` notebook to better illustrate how search works.

## Contacts and licence

Contacts :
- **Mail** : `yui-mhcp@tutanota.com`
- **[Discord](https://discord.com)** : yui0732

### Terms of use

The goal of these projects is to support and advance education and research in Deep Learning technology. To facilitate this, all associated code is made available under the [GNU Affero General Public License (AGPL) v3](AGPLv3.licence), supplemented by a clause that prohibits commercial use (cf the [LICENCE](LICENCE) file).

These projects are released as "free software", allowing you to freely use, modify, deploy, and share the software, provided you adhere to the terms of the license. While the software is freely available, it is not public domain and retains copyright protection. The license conditions are designed to ensure that every user can utilize and modify any version of the code for their own educational and research projects.

If you wish to use this project in a proprietary commercial endeavor, you must obtain a separate license. For further details on this process, please contact me directly.

For my protection, it is important to note that all projects are available on an "As Is" basis, without any warranties or conditions of any kind, either explicit or implied. However, do not hesitate to report issues on the repository's project, or make a Pull Request to solve it :smile: 

### Citation

If you find this project useful in your work, please add this citation to give it more visibility ! :yum:

```
@misc{yui-mhcp
    author  = {yui},
    title   = {A Deep Learning projects centralization},
    year    = {2021},
    publisher   = {GitHub},
    howpublished    = {\url{https://github.com/yui-mhcp}}
}
```

## Notes and references 

Github : 
- [NVIDIA's project](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechRecognition/Jasper) : original Jasper code.
- [NVIDIA's NeMo project](https://github.com/NVIDIA/NeMo) : provides a a pytorch implementation of the `Conformer` and `RNN-T` models.
- [Automatic Speech Recognition project](https://github.com/rolczynski/Automatic-Speech-Recognition) : DeepSpeech2 implementation. 
- [OpenAI's Whisper](https://github.com/openai/whisper) : the official OpenAI's implementation of Whisper (in pytorch).

Papers :
- [Deep Speech 2: End-to-End Speech Recognition in English and Mandarin](https://www.paperswithcode.com/paper/deep-speech-2-end-to-end-speech-recognition) : original DeepSpeech2 paper
- [Jasper: An End-to-End Convolutional Neural Acoustic Model](https://www.paperswithcode.com/paper/jasper-an-end-to-end-convolutional-neural) : the original Jasper paper
- [Speech-Transformer: A No-Recurrence Sequence-to-Sequence Model for Speech Recognition](https://ieeexplore.ieee.org/document/8462506) : original SpeechTransformer paper
- [A technique for computer detection and correction of spelling errors](https://dl.acm.org/doi/10.1145/363958.363994) : Levenshtein distance paper
- [RNN-T for Latency Controlled ASR WITH IMPROVED BEAM SEARCH](https://www.arxiv-vanity.com/papers/1911.01629/) : RNN-Transducer paper (maybe not the original one)
- [Conformer: Convolution-augmented Transformer for Speech Recognition](https://arxiv.org/abs/2005.08100) : Conformer original paper

Tutorials : 
- [Keras tutorial](https://keras.io/examples/audio/transformer_asr/) : tutorial on speech recognition with Transformers. 
- [Levenshtein distance computation](https://blog.paperspace.com/measuring-text-similarity-using-levenshtein-distance/) : a Step-by-Step computation (by hand) of the Levenshtein distance
- [NVIDIA NeMo project](https://developer.nvidia.com/nvidia-nemo) : main website for NVIDIA NeMo project, containing a lot of tutorials on NLP in general (ASR, TTS, ...). 
- [Comparing End-To-End Speech Recognition Architectures in 2021](https://www.assemblyai.com/blog/a-survey-on-end-to-end-speech-recognition-architectures-in-2021/) : good comparison between different STT architectures
