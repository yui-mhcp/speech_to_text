
# Copyright (C) 2022 yui-mhcp project's author. All rights reserved.
# Licenced under the Affero GPL v3 Licence (the "Licence").
# you may not use this file except in compliance with the License.
# See the "LICENCE" file at the root of the directory for the licence information.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" TF 2.0 Whisper model, compatible with the official model (https://github.com/openai/whisper). """

import os
import numpy as np
import tensorflow as tf

from custom_architectures.simple_models import HParamsConvBN, simple_cnn
from custom_architectures.transformers_arch.transformer_arch import *
from custom_architectures.transformers_arch.text_transformer_arch import TextTransformer
from custom_architectures.transformers_arch.gpt2_arch import GPT2, HParamsBaseGPT2

WHISPER_MODELS = {
    "tiny.en": "https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.en.pt",
    "tiny": "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt",
    "base.en": "https://openaipublic.azureedge.net/main/whisper/models/25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead/base.en.pt",
    "base": "https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt",
    "small.en": "https://openaipublic.azureedge.net/main/whisper/models/f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872/small.en.pt",
    "small": "https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt",
    "medium.en": "https://openaipublic.azureedge.net/main/whisper/models/d7440d1dc186f76616474e0ff0b3b6b879abc9d1a4926b7adfa41db2d497ab4f/medium.en.pt",
    "medium": "https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt",
    "large-v1": "https://openaipublic.azureedge.net/main/whisper/models/e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a/large-v1.pt",
    "large-v2": "https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt",
    "large": "https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt",
}

_whisper_loaded    = {}


HParamsWhisperEncoder = HParamsTransformerBlock(
    ** HParamsConvBN(
        n_conv  = 2,
        kernel_size = 3,
        padding = 'same',
        strides = [1, 2],
        bnorm   = 'never',
        activation  = 'gelu',
        final_activation = 'gelu'
    ),
    n_mel_channels  = 80,
    max_input_length    = 1500,
    normalize_output    = True,
    
    epsilon = 1e-5,
    mha_epsilon = 1e-5,
    mha_key_bias    = False,
    mha_mask_factor = -1.,
    
    normalize   = 'middle',
    mha_normalize   = False,
    mha_normalize_input = True,
    
    ffn_dim     = 2048,
    ffn_activation  = 'gelu'
)

HParamsWhisperDecoder   = HParamsBaseGPT2(
    use_encoder_attention   = True,
    use_causal_attention    = True,
    positional_offset   = 0,
    scale_embedding = False,
    
    normalize   = 'middle',
    mha_normalize   = False,
    enc_mha_normalize   = False,

    mha_mask_factor = -1.,
    enc_mha_mask_factor = -1.,
    
    mha_key_bias    = False,
    enc_mha_key_bias    = False,
    
    mha_normalize_input = True,
    enc_mha_normalize_input = True,
    enc_mha_epsilon = 1e-5,
    mha_epsilon = 1e-5,
    epsilon = 1e-5
)

def get_pos_embedding(max_length, embedding_dim, max_timescale = 10000, dtype = tf.float32):
    """ Returns sinusoids for positional embedding """
    assert embedding_dim % 2 == 0
    
    log_timescale_increment = tf.cast(np.log(max_timescale) / (embedding_dim // 2 - 1), dtype)

    inv_timescales  = tf.exp(-log_timescale_increment * tf.range(embedding_dim // 2, dtype = dtype))
    scaled_time = tf.range(max_length, dtype = dtype)[:, tf.newaxis] * inv_timescales[tf.newaxis, :]
    
    return tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis = 1)

class WhisperEncoder(TransformerBlock):
    default_params  = HParamsWhisperEncoder
    _attr_to_set    = TransformerBlock._attr_to_set + ['n_mel_channels', 'max_input_length']
    
    def __init__(self, n_mel_channels = 80, embedding_dim = 512, ** kwargs):
        super().__init__(
            n_mel_channels = n_mel_channels, embedding_dim = embedding_dim, ** kwargs
        )
    
    def _init_input_layers(self, ** kwargs):
        self.feature_extractor  = simple_cnn(** self.hparams(
            use_sequential  = False,
            input_shape     = (None, self.n_mel_channels),
            output_shape    = self.embedding_dim,
            conv_type   = 'conv1d',
            use_mask    = True,
            filters     = self.embedding_dim,
            final_activation    = self.hparams.activation,
            flatten     = False,
            dense_as_final  = False,
            name    = 'feature_extractor'
        ))
        
        self.pos_encoding   = tf.expand_dims(get_pos_embedding(
            self.max_input_length, self.embedding_dim
        ), axis = 0)

    @property
    def input_signature(self):
        return tf.TensorSpec(shape = (None, None, self.n_mel_channels), dtype = tf.float32)
    
    @property
    def dummy_inputs(self):
        return tf.random.normal((1, 128, self.n_mel_channels))

    def prepare_input(self, inputs, training = False, mask = None, ** kwargs):
        embedded = self.feature_extractor(inputs, training = training, mask = mask)
        mask     = tf.reshape(
            embedded._keras_mask, [tf.shape(embedded)[0], 1, 1, tf.shape(embedded)[1]]
        )

        embedded = embedded + self.pos_encoding[:, :tf.shape(embedded)[1]]
        embedded._keras_mask = mask
        
        return embedded

    def transfer_weights(self, pretrained, ** kwargs):
        from models.weights_converter import _transformer_patterns, name_based_partial_transfer_learning

        if isinstance(pretrained, dict):
            pretrained = {k : v for k, v in pretrained.items() if k.startswith('encoder')}

        return name_based_partial_transfer_learning(
            self, pretrained, skip_root = False, patterns = {
                ** _transformer_patterns, 'blocks/' : 'layer_', '_ln' : '_norm',
                'ffn/0' : 'ffn/dense_1', 'ffn/2' : 'ffn/dense_2', 'mha_norm' : 'mha/norm_input'
            }, ** kwargs
        )
        
    @classmethod
    def from_pretrained(cls,
                        pretrained_name = 'medium',
                        pretrained      = None,
                        tqdm    = lambda x: x,
                        ** kwargs
                       ):
        pretrained = load_whisper(pretrained_name, pretrained = pretrained)
        pretrained, config = pretrained['model_state_dict'], pretrained['dims']
        
        config = cls.default_params(
            n_mel_channels  = config['n_mels'],
            embedding_dim   = config['n_audio_state'],
            max_input_length    = config['n_audio_ctx'],
            
            num_layers  = config['n_audio_layer'],
            mha_num_heads   = config['n_audio_head']
        )

        instance = cls(** config(** kwargs))
        instance._build()

        instance.transfer_weights(pretrained, tqdm = tqdm, ** kwargs)

        return instance

class WhisperDecoder(GPT2):
    default_params  = HParamsWhisperDecoder

    def transfer_weights(self, pretrained, ** kwargs):
        from models.weights_converter import _transformer_patterns, name_based_partial_transfer_learning

        if isinstance(pretrained, dict):
            pretrained = {k : v for k, v in pretrained.items() if k.startswith('decoder')}
        
        return name_based_partial_transfer_learning(
            self, pretrained, skip_root = False, patterns = {
                ** _transformer_patterns, 'blocks/' : 'layer_', '_ln' : '_norm',
                'ffn/0' : 'ffn/dense_1', 'ffn/2' : 'ffn/dense_2',
                '(attn|mha)_norm' : 'mha/norm_input', 'cross_' : 'enc_', 'enc_attn' : 'enc_mha'
            }, ** kwargs
        )
        
    @classmethod
    def from_pretrained(cls,
                        pretrained_name = 'medium',
                        pretrained      = None,
                        tqdm    = lambda x: x,
                        ** kwargs
                       ):
        pretrained = load_whisper(pretrained_name, pretrained = pretrained)
        pretrained, config = pretrained['model_state_dict'], pretrained['dims']
        
        config = cls.default_params(
            vocab_size  = config['n_vocab'],
            embedding_dim   = config['n_text_state'],
            max_input_length    = config['n_text_ctx'],
            
            num_layers  = config['n_text_layer'],
            mha_num_heads   = config['n_text_head']
        )

        instance = cls(** config(** kwargs))
        instance._build()

        instance.transfer_weights(pretrained, tqdm = tqdm, ** kwargs)

        return instance

class Whisper(TextTransformer):
    encoder_class   = WhisperEncoder
    decoder_class   = WhisperDecoder
    
    _shared_keys    = TextTransformer._shared_keys + ['n_mel_channels']
    
    @classmethod
    def from_pretrained(cls,
                        pretrained_name = 'medium',
                        pretrained      = None,
                        tqdm    = lambda x: x,
                        ** kwargs
                       ):
        pretrained = load_whisper(pretrained_name, pretrained = pretrained)
        pretrained, config = pretrained['model_state_dict'], pretrained['dims']
        
        config = cls.default_params(
            vocab_size      = config['n_vocab'],
            n_mel_channels  = config['n_mels'],
            
            encoder_embedding_dim   = config['n_audio_state'],
            encoder_max_input_length    = config['n_audio_ctx'],
            encoder_num_layers  = config['n_audio_layer'],
            encoder_mha_num_heads   = config['n_audio_head'],

            decoder_embedding_dim   = config['n_text_state'],
            decoder_max_input_length    = config['n_text_ctx'],
            decoder_num_layers  = config['n_text_layer'],
            decoder_mha_num_heads   = config['n_text_head']
        )

        instance = cls(** config(** kwargs))
        instance._build()

        instance.encoder.transfer_weights(pretrained, tqdm = tqdm, ** kwargs)
        instance.decoder.transfer_weights(pretrained, tqdm = tqdm, ** kwargs)

        return instance

def load_whisper(pretrained_name = 'medium', pretrained = None, ** kwargs):
    global _whisper_loaded
    
    if isinstance(pretrained, str): pretrained_name, pretrained = pretrained, None
    
    if pretrained is None:
        from utils.file_utils import download_file
        from models import _pretrained_models_folder
        
        if pretrained_name not in _whisper_loaded:
            import torch

            if pretrained_name not in WHISPER_MODELS:
                raise ValueError('Unknown pretrained Whisper model !\n  Accepted : {}\n  Got : {}'.format(
                    tuple(WHISPER_MODELS.keys()), pretrained_name
                ))

            filename = download_file(
                WHISPER_MODELS[pretrained_name],
                directory = os.path.join(_pretrained_models_folder, 'pretrained_weights')
            )

            if filename is None:
                raise RuntimeError('filename is None, an error has occured while loading it')

            pretrained  = torch.load(filename, map_location = 'cpu')
            if not isinstance(pretrained, dict): pretrained = pretrained.state_dict()
            _whisper_loaded[pretrained_name] = pretrained
        
        pretrained = _whisper_loaded[pretrained_name]
    
    state_dict = pretrained if isinstance(pretrained, dict) else {
        'dims' : pretrained.dims.__dict__, 'model_state_dict' : pretrained.state_dict()
    }
    
    return state_dict

_whisper_classes    = {
    'WhisperEncoder'    : WhisperEncoder,
    'WhisperDecoder'    : WhisperDecoder,
    'Whisper'       : Whisper
}

custom_functions    = {
    ** _whisper_classes
}

custom_objects  = {
    ** _whisper_classes
}
_encoders   = {'Whisper' : WhisperEncoder}
_decoders   = {'Whisper' : WhisperDecoder}
_transformers   = {'Whisper'    : Whisper}