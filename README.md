# :yum: Speech To Text (STT)

Check the [CHANGELOG](https://github.com/yui-mhcp/yui-mhcp/blob/main/CHANGELOG.md) file to have a global overview of the latest modifications ! :yum:

**IMPORTANT NOTE** : currently, only the `Whisper` models have been refactored. The other architectures (such as `Jasper` and `DeepSpeech`) may not properly work, especially in their `predict` method. 

## Project structure

```bash
.
├── custom_architectures
│   ├── transformers_arch
│   │   ├── conformer_arch.py
│   │   └── transformer_stt_arch.py
│   ├── deep_speech_2_arch.py
│   ├── jasper_arch.py
│   ├── rnnt_arch.py                    : RNN-T abstract architecture class
│   └── transducer_generation_utils.py  : RNN-T inference script
├── custom_layers
├── custom_train_objects
├── datasets
├── hparams
├── loggers
├── models
│   ├── stt
│   │   ├── base_stt.py             : abstract class for STT models
│   │   ├── conformer_transducer.py : Conformer Transducer main class
│   │   ├── deep_speech.py          : Deep Speech 2 main class
│   │   ├── jasper.py               : Jasper main class
│   │   ├── transformer_stt.py      : Transformer STT main class
│   │   └── whisper.py              : Whisper main class
├── pretrained_models
│   └── pretrained_weights          : where to put the jasper / deep speech pretrained weights
├── unitest
├── utils
├── example_training.ipynb
└── speech_to_text.ipynb
```

Check [the main project](https://github.com/yui-mhcp/base_dl_project) for more information about the unextended modules / structure / main classes. 

## Available features

- **Speech-To-Text** (module `models.stt`) :

| Feature   | Fuction / class   | Description |
| :-------- | :---------------- | :---------- |
| Speech-To-Text    | `stt`             | Perform `STT` on audio / video      |
| search            | `search`          | Search a word on audio / video and display timestamps |
| stream            | `{model}.stream`  | Speech-To-Text on your microphone (experimental)  |

You can check the `speech_to_text` notebook for a concrete demonstration

**Note** : most models (except `Whisper`) are not performant enough for real *transcription / subtitles generation* but the search feature works quite well. The `Conformer-Transducers` models tend to be good enough for transcription but it is currently an experimental feature which will be improved (by fine-tuning them or adding the Beam Search inference).

## Available models

### Model architectures

Available architectures : 
- `CTC decoders` :
    - [DeepSpeech2](https://www.paperswithcode.com/paper/deep-speech-2-end-to-end-speech-recognition)
    - [Jasper](https://www.paperswithcode.com/paper/jasper-an-end-to-end-convolutional-neural)
    - [Conformer](https://arxiv.org/pdf/2005.08100.pdf)
- `Generative models` :
    - [Speech Transformer](https://ieeexplore.ieee.org/document/8462506)
    - [RNN Transducer (RNN-T)](https://www.arxiv-vanity.com/papers/1911.01629/)
    - [Conformer Transducer](https://www.arxiv-vanity.com/papers/1911.01629/) : `RNN-T` based model with the `Conformer` architecture as encoder.
    - [Whisper](https://github.com/openai/whisper) : OpenAI's Whisper multilingual STT model.

### Model weights

| Language  | Dataset   | Architecture  | Trainer   | Weights   |
| :-------: | :-------: | :-----------: | :-------: | :-------: |
| `en`      | `LibriSpeech` | `Jasper`  | [NVIDIA](https://github.com/NVIDIA)   | [Google Drive](https://drive.google.com/file/d/1JViFiy-JZ8VYTlaZPVDMZfg0qWcY5-U8/view?usp=sharing)\*  |
| `fr`      | `SIWIS`, `VoxForge`, `Common Voice`   | `Jasper`  | [me](https://github.com/yui-mhcp) | [Google Drive](https://drive.google.com/file/d/1R9lXaEj4etAyyfy7r3tYNnO5FPwQ7RXS/view?usp=sharing)  |
| `en`      | Many datasets | `ConformerTransducer` | [NVIDIA](https://github.com/NVIDIA) | [Google Drive](https://drive.google.com/file/d/1OpWBvkERK9IVQ1BZPsBHOs46tWn_H2mZ/view?usp=sharing)  |
| `fr`      | Many datasets | `ConformerTransducer` | [NVIDIA](https://github.com/NVIDIA) | [Google Drive](https://drive.google.com/file/d/1cftQBZEmKL-2fLKpmboWTXIvgFgccsPf/view?usp=sharing)  |


`ConformerTransducer` models come from the [NVIDIA NeMo project](https://github.com/NVIDIA/NeMo) but are converted in tensorflow weights. To re-create them from NeMo, you have to clone the NeMo project and use the `ConformerTransducer.from_nemo_pretrained` method.

**WARNING** : `Jasper` model weights are 3Go files !

Models must be unzipped in the `pretrained_models/` directory !

\* This file is a `.h5` weights file. You have to put it in `pretrained_models/pretrained_weights/` folder and check the code in `example_training` notebook ;)

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
- [x] Add Beam-Search decoding (for acoustic models)
- [x] Add multilingual model support (`Whisper`)
- [x] Implement `Transformer`-based STT models (such as `TransformerSTT` and `Conformer`) (in progress)
- [x] Implement a `RNN-T` based STT model (such as `Conformer Transducer`)
- [x] Add `producer-consumer` based inference
- [x] Add `producer-consumer` based streaming (currently only for `conformer-transducer`)
- [x] Add streaming support (experimental)
- [ ] Add Beam Search inference for RNN-T models
- [ ] Add pipeline-based prediction for mic streaming
- [ ] Try to fix the small difference in output scores between the original pytorch whisper model and my tensorflow implementation (the difference seems to come from the `Conv` functions which differ between pytorch and tensorflow)
- [ ] Convert `Whisper` pretrained models from the `transformers` hub (currently, only the official `Whisper` weights are supported)

## Search and partial alignment

Models are not good enough for real *transcription* so we cannot *simply* search a word in the transcription produced. To solve this problem, I have decided to use the **Levenshtein distance** (or **edit distance**) metric which allows to search for alignments. 

For instance, searching *cat* in *th ct* will not find *cat* but you can see that *ct* has only 1 error (the missing *a*). 

The Levenshtein distance will produce alignment between *cat* and *ct* with a distance matrix (I put a link for step-by-step computing of this matrix) : 

|   |   | c | t |
|:-:|:-:|:-:|:-:|
|   | 0 | 1 | 2 |
| c | 1 | 0 | 1 |
| a | 2 | 1 | 1 |
| t | 3 | 2 | 1 |

As you can see, the bottom-right value is 1 which represents the total distance between the `truth` (*ct*) and the `hypothesis` (*cat*). 

The value at index **i, j** is the minimum between :
- matrix[i-1][j]    + deletion cost of caracter i (in hypothesis)
- matrix[i-1][j-1]  + replacement cost of caracter i (in hypothesis) and caracter j (of truth) (equal to 0 if both are same caracter)
- matrix[i][j-1]    + insertion cost of caracter j (of hypothesis)

Note : to simplify example, I put insertion / deletion costs equal to 1 but, in practice, cost to add / remove spaces / punctuation is 0 (and some other improvments). 

The objective is now to align *cat* at all position of `truth` (*the ct*). For this purpose, you can put the 1st line to 0 so it will try to align *cat* at each position without penalizing the position of the alignment. 

|   |   | t | h | e |   | c | t |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|   | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| c | 1 | 1 | 1 | 1 | 1 | 0 | 1 |
| a | 2 | 2 | 2 | 2 | 2 | 1 | 1 |
| t | 3 | 2 | 3 | 3 | 3 | 2 | 1 |

With this simple trick, you can see that *cat* has a distance of 1 when the alignment is finishing at the last index of `truth` ! It is how the search is implemented so you can see 1 error out of 3 caracters which will give a probability of 66%.

Note : as you can see, scores will be less influenced (so more pertinent) if the searched word is longer. 

There is an example in the `speech_to_text` notebook to better illustrate how search works.

## Contacts and licence

You can contact [me](https://github.com/yui-mhcp) at yui-mhcp@tutanota.com or on [discord](https://discord.com) at `yui#0732`

The objective of these projects is to facilitate the development and deployment of useful application using Deep Learning for solving real-world problems and helping people. 
For this purpose, all the code is under the [Affero GPL (AGPL) v3 licence](LICENCE)

All my projects are "free software", meaning that you can use, modify, deploy and distribute them on a free basis, in compliance with the Licence. They are not in the public domain and are copyrighted, there exist some conditions on the distribution but their objective is to make sure that everyone is able to use and share any modified version of these projects. 

Furthermore, if you want to use any project in a closed-source project, or in a commercial project, you will need to obtain another Licence. Please contact me for more information. 

For my protection, it is important to note that all projects are available on an "As Is" basis, without any warranties or conditions of any kind, either explicit or implied. However, do not hesitate to report issues on the repository's project or make a Pull Request to solve it :smile: 

If you use this project in your work, please add this citation to give it more visibility ! :yum:

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
- [NVIDIA's project](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechRecognition/Jasper) : repository where pretrained weights come from (I converted pytorch checkpoint to tensorflow checkpoint using my `weights_converter` script).
- [NVIDIA's NeMo project](https://github.com/NVIDIA/NeMo) : the Conformer and the RNN-T implementations are inspired from this github (but converted to tensorflow).
- [Automatic Speech Recognition project](https://github.com/rolczynski/Automatic-Speech-Recognition) : project where pretrained DeepSpeech2 weights come from. 
- [OpenAI's Whisper](https://github.com/openai/whisper) : the official OpenAI's implementation of Whisper (in pytorch).

Papers :
- [1] [Deep Speech 2: End-to-End Speech Recognition in English and Mandarin](https://www.paperswithcode.com/paper/deep-speech-2-end-to-end-speech-recognition) : original DeepSpeech2 paper
- [2] [Jasper: An End-to-End Convolutional Neural Acoustic Model](https://www.paperswithcode.com/paper/jasper-an-end-to-end-convolutional-neural) : the original Jasper paper
- [3] [Speech-Transformer: A No-Recurrence Sequence-to-Sequence Model for Speech Recognition](https://ieeexplore.ieee.org/document/8462506) : original SpeechTransformer paper
- [4] [A technique for computer detection and correction of spelling errors](https://dl.acm.org/doi/10.1145/363958.363994) : Levenshtein distance paper
- [5] [RNN-T for Latency Controlled ASR WITH IMPROVED BEAM SEARCH](https://www.arxiv-vanity.com/papers/1911.01629/) : RNN-Transducer paper (maybe not the original one)
- [6] [Conformer: Convolution-augmented Transformer for Speech Recognition](https://arxiv.org/abs/2005.08100) : Conformer original paper

Tutorials : 
- [Keras tutorial](https://keras.io/examples/audio/transformer_asr/) : tutorial on speech recognition with Transformers. The model is implemented in this repo but I did not achieve to reproduce results on a french dataset
- [Levenshtein distance computation](https://blog.paperspace.com/measuring-text-similarity-using-levenshtein-distance/) : a Step-by-Step computation (by hand) of the Levenshtein distance
- [NVIDIA NeMo project](https://developer.nvidia.com/nvidia-nemo) : main website for NVIDIA NeMo project, containing a lot of tutorials on NLP in general (ASR, TTS, ...). 
- [Comparing End-To-End Speech Recognition Architectures in 2021](https://www.assemblyai.com/blog/a-survey-on-end-to-end-speech-recognition-architectures-in-2021/) : good comparison between different STT architectures
