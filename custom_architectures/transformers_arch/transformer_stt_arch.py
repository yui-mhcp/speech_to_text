
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

""" TF 2.0 BART model, compatible with the `transformers`' model. """

import numpy as np
import tensorflow as tf

from custom_architectures.transformers_arch.transformer_arch import *
from custom_architectures.transformers_arch.text_transformer_arch import *
from custom_architectures.transformers_arch.bart_arch import BartDecoder, HParamsBartDecoder

_conv_kwargs    = {
    'filters'       : 1024,
    'kernel_size'   : 5,
    'strides'       : 2
}

HParamsTransformerSTTFeatureExtractor   = HParams(
    ** _conv_kwargs,
    embedding_dim   = -1,
    n_conv  = 2,
    conv_activation = 'relu',
    conv_drop_rate  = 0.25
)

HParamsTransformerSTTEncoder    = HParamsTransformerEncoder(
    ** HParamsTransformerSTTFeatureExtractor,
    n_mel_channels  = -1,
    max_input_length    = 6000,
    positional_embedding_type   = 'embedding',
    positional_offset   = 2,
    scale_embedding = False
)

def _get_var(_vars, i):
    if callable(_vars) and _vars.__class__.__name__ == 'function': return _vars(i)
    elif isinstance(_vars, list): return _vars[i]
    else: return _vars

class TransformerSTTFeatureExtractor(tf.keras.Model):
    def __init__(self, embedding_dim, name = 'feature_extractor', ** kwargs):
        super().__init__(name = name)
        
        self.hparams = HParamsTransformerSTTFeatureExtractor.extract({
            ** kwargs, 'embedding_dim' : embedding_dim
        })
        
        self.subsampling_factor = 1
        self.paddings   = []
        
        self.conv_layers    = []
        self.act_layers     = []
        for i in range(self.hparams.n_conv):
            conv_config = {
                k : _get_var(self.hparams[k], i) for k in _conv_kwargs.keys()
            }
            if conv_config['filters'] == -1 or i == self.hparams.n_conv - 1:
                conv_config['filters'] = self.hparams.embedding_dim * 2
            
            self.subsampling_factor *= conv_config['strides']
            self.paddings.append(tf.constant(conv_config['kernel_size'] // 2, dtype = tf.int32))

            self.conv_layers.append(tf.keras.layers.Conv1D(
                ** conv_config, padding = 'valid', name = 'conv_{}'.format(i)
            ))
            self.act_layers.append(get_activation(_get_var(self.hparams.conv_activation, i)))
            
        self.dropout = tf.keras.layers.Dropout(self.hparams.conv_drop_rate)
    
    @timer
    def call(self, inputs, input_length = None, training = False, ** kwargs):
        features = inputs
        for layer, act, pad in zip(self.conv_layers, self.act_layers, self.paddings):
            features = tf.pad(features, [(0, 0), (pad, pad), (0, 0)])
            
            features = layer(features, training = training)
            if act is not None: features = act(features)
            features = self.dropout(features, training = training)
        
        if input_length is not None:
            input_length = input_length / self.subsampling_factor
        
        return features, input_length

    def get_config(self):
        return self.hparams.get_config()
    
    @classmethod
    def from_config(cls, config, custom_objects = None):
        return cls(** config)
    
class TransformerSTTEncoder(TransformerBlock):
    default_params  = HParamsTransformerSTTEncoder
    _attr_to_set    = TransformerBlock._attr_to_set + ['positional_offset']

    def __init__(self, n_mel_channels, embedding_dim, max_input_length, ** kwargs):
        super().__init__(
            n_mel_channels = n_mel_channels, embedding_dim = embedding_dim,
            max_input_length = max_input_length, ** kwargs
        )
        
        self.embedding_factor = tf.math.sqrt(float(embedding_dim) if self.hparams.scale_embedding else 1.)

        self.final_norm = tf.keras.layers.LayerNormalization(epsilon = self.hparams.epsilon)
    
    def _init_input_layers(self, ** kwargs):
        self.feature_extraction_layer  = TransformerSTTFeatureExtractor(** self.hparams)
        
        if self.hparams.positional_embedding_type == 'embedding':
            positional_embedding    = FasterEmbedding(
                self.max_input_length, self.embedding_dim, name = "pos_embeddings"
            )
        elif self.hparams.positional_embedding_type == 'sin':
            raise NotImplementedError()
        self.pos_embedding_layer    = positional_embedding
    
    @property
    def max_input_length(self):
        return self.hparams.max_input_length + self.positional_offset
    
    @property
    def dummy_inputs(self):
        batch_size, seq_len = 2, 128
        audio = tf.random.normal(
            [batch_size, seq_len, self.hparams.n_mel_channels], dtype = tf.float32
        )
        audio_length = tf.fill([batch_size, 1], seq_len)
        
        return [audio, audio_length]
    
    @timer
    def embed_positions(self, position_ids, seq_len, positional_offset, debug = False, ** kwargs):
        if position_ids is None:
            position_ids = tf.range(seq_len)

            position_ids = tf.expand_dims(position_ids, axis = 0)
            if positional_offset == -1: positional_offset = self.positional_offset
            if positional_offset > 0:
                position_ids = position_ids + positional_offset
        
        if debug:
            tf.print("Position ids :", position_ids)
        
        return self.pos_embedding_layer(position_ids)
        
    def compute_output(self, output, training = False, mask = None, ** kwargs):
        return self.final_norm(output, training = training)
    
    @timer
    def call(self,
             inputs,
             input_length   = None,
             position_ids   = None,
             mask       = None,
             training   = False,
             positional_offset  = -1,
             ** kwargs
            ):
        audio = inputs
        if isinstance(inputs, (list, tuple)):
            audio, input_length = inputs[:2]
            if len(inputs) > 2: position_ids = inputs[2]
        
        features, input_length = self.feature_extraction_layer(
            audio, input_length = input_length, training = training, mask = mask, ** kwargs
        )
        
        pos_embedded    = self.embed_positions(
            position_ids, tf.shape(features)[1], positional_offset, ** kwargs
        )
        
        features    = features * self.embedding_factor
        
        features    = features + pos_embedded
        
        outputs = super().call(
            features, input_length = input_length, mask = None, training = training, ** kwargs
        )
        if not isinstance(outputs, (list, tuple, TransformerOutput)): outputs = (outputs, )
        decoder_outputs = outputs[0]

        logits = self.compute_output(
            decoder_outputs, training = training, mask = mask, ** kwargs
        )
        
        if isinstance(outputs, TransformerOutput):
            return TransformerOutput(logits, * outputs[1:])
        elif len(outputs) > 1:
            return (logits, ) + outputs[1:]
        return logits
    
    def transfer_weights(self, pretrained):
        from models.weights_converter import partial_transfer_learning, get_pt_variables
        offset, n_enc_layer_weights = self.hparams.n_conv * 2 + 1, 16
        
        weights = get_pt_variables(pretrained.encoder)
        weights[offset - 1] = weights[offset - 1].T

        for i in range(pretrained.config.encoder_layers):
            weights[i * n_enc_layer_weights + offset], weights[i * n_enc_layer_weights + offset + 2] = (
                weights[i * n_enc_layer_weights + offset + 2], weights[i * n_enc_layer_weights + offset]
            )
            weights[i * n_enc_layer_weights + offset + 1], weights[i * n_enc_layer_weights + offset + 3] = (
                weights[i * n_enc_layer_weights + offset + 3], weights[i * n_enc_layer_weights + offset + 1]
            )
        # Add shared embeddings weights to the list
        partial_transfer_learning(self, weights)

    @classmethod
    def from_pretrained(cls,
                        pretrained_name = 'facebook/s2t-small-librispeech-asr',
                        pretrained_task = 'generation',
                        pretrained      = None,
                        ** kwargs
                       ):
        if pretrained is None:
            pretrained = transformers_stt(pretrained_name, pretrained_task)

        config = HParamsTransformerSTTEncoder(
            n_mel_channels  = pretrained.config.input_feat_per_channel,
            embedding_dim   = pretrained.config.d_model,
            max_input_length    = pretrained.config.max_source_positions,
            scale_embedding = pretrained.config.scale_embedding,
            normalize   = 'middle',
            epsilon = 1e-5,

            n_conv  = pretrained.config.num_conv_layers,
            filters = pretrained.config.conv_channels,
            kernel_size = pretrained.config.conv_kernel_sizes,
            padding = 'valid',
            strides = 2,
            conv_activation = 'glu',

            num_layers  = pretrained.config.encoder_layers,
            ffn_dim     = pretrained.config.encoder_ffn_dim,
            ffn_activation  = pretrained.config.activation_function,
            mha_num_heads   = pretrained.config.encoder_attention_heads,
            mha_normalize_input = True,
            mha_normalize   = False
        )
        
        instance = cls(** config(** kwargs))
        instance._build()

        instance.transfer_weights(pretrained)
        
        return instance


class TransformerSTT(TextTransformer):
    encoder_class   = TransformerSTTEncoder
    decoder_class   = BartDecoder

    @classmethod
    def from_pretrained(cls,
                        pretrained_name = 'facebook/bart-large',
                        pretrained_task = 'generation', 
                        pretrained      = None,
                        ** kwargs
                       ):
        if pretrained is None:
            with tf.device('cpu') as d:
                pretrained = transformers_bart(pretrained_name, pretrained_task)

        config = HParamsBart(
            vocab_size      = pretrained.config.vocab_size,
            embedding_dim   = pretrained.config.d_model,
            max_input_length    = pretrained.config.max_position_embeddings,
            positional_offset   = 2,
            scale_embedding = False,
            epsilon     = 1e-5,
            sos_token   = 0,
            eos_token   = 2,

            encoder_num_layers  = pretrained.config.encoder_layers,
            encoder_ffn_dim     = pretrained.config.encoder_ffn_dim,
            encoder_ffn_activation  = pretrained.config.activation_function,
            encoder_mha_num_heads   = pretrained.config.encoder_attention_heads,

            decoder_num_layers  = pretrained.config.decoder_layers,
            decoder_ffn_dim     = pretrained.config.decoder_ffn_dim,
            decoder_ffn_activation  = pretrained.config.activation_function,
            decoder_mha_num_heads   = pretrained.config.decoder_attention_heads,
            decoder_enc_mha_num_heads   = pretrained.config.decoder_attention_heads
        )
        
        instance = cls(** config(** kwargs))
        instance._build()
        
        instance.encoder.transfer_weights(pretrained)
        instance.decoder.transfer_weights(pretrained)
        
        return instance

def transformers_stt(name = 'facebook/s2t-small-librispeech-asr', task = 'generation'):
    import transformers
    if task == 'generation':
        return transformers.Speech2TextForConditionalGeneration.from_pretrained(name)
    else:
        raise ValueError("Unknown task !\n  Accepted : {}\n  Got : {}".format(
            tuple(_transformers_pretrained_task.keys()), task
        ))

_transformers_stt_classes   = {
    'TransformerSTTFeatureExtractor'    : TransformerSTTFeatureExtractor,
    'TransformerSTTEncoder' : TransformerSTTEncoder,
    'BartDecoder'   : BartDecoder,
    'TransformerSTT'    : TransformerSTT
}
        
custom_functions    = {
    ** _transformers_stt_classes,
    'transformers_stt'  : transformers_stt
}

custom_objects  = {
    ** _transformers_stt_classes
}