# :yum: Speech To Text (STT)

## Project structure

```bash
├── custom_architectures/   : custom architectures
├── custom_layers/          : custom layers
├── custom_train_objects/   : custom objects for training
│   ├── callbacks/          : custom callbacks
│   ├── generators/         : custom data generators
│   ├── losses/             : custom losses
│   ├── optimizers/         : custom optimizers / lr schedulers
├── datasets/               : utilities for dataset loading / processing
│   ├── custom_datasets/    : where to save custom datasets processing
├── hparams/                : utility class to define modulable hyper-parameters
├── models/                 : main `BaseModel` subclasses directory
│   ├── stt/                : directory for Speech-To-Text models
├── pretrained_models/      : saving directory for pretrained models
└── utils/

```

See [my data_processing repo](https://github.com/yui-mhcp/data_processing) for more information on the `utils` module and `data processing` features.

See [my base project](https://github.com/yui-mhcp/base_dl_project) for more information on the `BaseModel` class, supported datasets, project extension, ...

## Available features

- **Speech-To-Text** (module `models.stt`) :

| Feature   | Fuction / class   | Description |
| :-------- | :---------------- | :---------- |
| Speech-To-Text    | `stt`             | Perform `STT` on audio / video      |
| search            | `search`          | Search a word on audio / video and display timestamps |

You can check the `speech_to_text` notebook for a concrete demonstration

**Note** : models are not performant enough for real *transcription / subtitles generation* but the search feature works quite well. 

## Available models

### Model architectures

Available architectures : 
- `CTC decoders` :
    - [DeepSpeech2](https://www.paperswithcode.com/paper/deep-speech-2-end-to-end-speech-recognition)
    - [Jasper](https://www.paperswithcode.com/paper/jasper-an-end-to-end-convolutional-neural)
- `Generative models` :
    - [Speech Transformer](https://ieeexplore.ieee.org/document/8462506)

### Model weights

| Classes   | Dataset   | Architecture  | Trainer   | Weights   |
| :-------: | :-------: | :-----------: | :-------: | :-------: |
| `en`      | `LibriSpeech` | `Jasper`      | [NVIDIA](https://github.com/NVIDIA)   | [Google Drive](https://drive.google.com/drive/folders/1mAUv_dKK50P0-ffYQgtEWnNc9gYk4ZwX?usp=sharing)  |

Models must be unzipped in the `pretrained_models/` directory !

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
- [ ] Add new languages support

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

Note : as you can see, score will be less influenced (so more pertinent) if the searched word is longer. 

## Contacts and licence

You can contact [me](https://github.com/yui-mhcp) at yui-mhcp@tutanota.com or on [discord](https://discord.com) at `yui#0732`

The objective of these projects is to facilitate the development and deployment of useful application using Deep Learning for solving real-world problems and helping people. 
For this purpose, all the code is under the [GNU GPL v3 licence](LICENCE)

Furthermore, you **cannot** use any of these projects for commercial purpose without my permission. You can use, modify, distribute and use any of my projects for production as long as you respect the terms of the [licence](LICENCE) and use it for non-commercial purposes (i.e. free applications / research). 

If you use this project in your work, please cite this project to give it more visibility ! :smile:

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
- [Automatic Speech Recognition project](https://github.com/rolczynski/Automatic-Speech-Recognition) : project where pretrained DeepSpeech2 weights come from. 

Papers :
- [1] [Deep Speech 2: End-to-End Speech Recognition in English and Mandarin](https://www.paperswithcode.com/paper/deep-speech-2-end-to-end-speech-recognition) : original DeepSpeech2 paper
- [2] [Jasper: An End-to-End Convolutional Neural Acoustic Model](https://www.paperswithcode.com/paper/jasper-an-end-to-end-convolutional-neural) : the original Jasper paper
- [3] [Speech-Transformer: A No-Recurrence Sequence-to-Sequence Model for Speech Recognition](https://ieeexplore.ieee.org/document/8462506) : original SpeechTransformer paper
- [4] [A technique for computer detection and correction of spelling errors](https://dl.acm.org/doi/10.1145/363958.363994) : Levenshtein distance paper

Tutorials : 
- [Keras tutorial](https://keras.io/examples/audio/transformer_asr/) : tutorial on speech recognition with Transformers. The model is implemented in this repo but I did not achieve to reproduce results on a french dataset
- [Levenshtein distance computation](https://blog.paperspace.com/measuring-text-similarity-using-levenshtein-distance/) : a Step-by-Step computation (by hand) of the Levenshtein distance
