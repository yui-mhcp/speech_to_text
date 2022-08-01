

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

""" TF 2.0 Conformer model, compatible with the `NeMo`'s model. """

import enum
import math
import tensorflow as tf

from hparams import HParams
from custom_layers import (
    get_activation, glu,
    HParamsMHA, MultiHeadAttention, RelPosMultiHeadAttention,
    FasterEmbedding, PositionalEmbedding, RelPositionalEmbedding
)
from custom_architectures.transformers_arch.transformer_arch import (
    _base_enc_dec_kwargs, build_mask, format_output, FeedForwardNetwork, TransformerBlock
)
from custom_architectures.rnnt_arch import RNNT, HParamsRNNT, HParamsRNNTDecoder, HParamsRNNTJointNet
from utils.generic_utils import get_enum_item
from loggers import timer

class ConcatMode(enum.IntEnum):
    CONCAT  = 0
    ADD     = 1
    SUB     = 2
    MUL     = 3
    DIV     = 4

class ConcatPos(enum.IntEnum):
    AFTER_SAMPLING  = 0
    BEFORE_POS      = 0
    BEFORE_SCALING  = 1
    AFTER_SCALING   = 2
    END             = 3

_conv_kwargs    = {'filters' : 256, 'kernel_size' : 3, 'strides' : 2}

HParamsConformerSubsampler   = HParams(
    ** {'subsampler_{}'.format(k) : v for k, v in _conv_kwargs.items()},
    embedding_dim   = -1,
    n_mel_channels  = -1,
    n_conv  = 2,
    subsampler_activation = 'relu',
    subsampler_drop_rate  = 0.25
)

HParamsConformerConvolution = HParams(
    embedding_dim   = -1,
    kernel_size = 31,
    conv_activation = 'swish',
    conv_norm_type  = 'batch_norm',
    epsilon = 1e-5
)

HParamsConformerLayer   = HParams(
    ** HParamsMHA.get_config(add_prefix = 'mha'),
    ** HParamsConformerConvolution,
    fc_factor   = 0.5,
    drop_rate   = 0.1,
    norm_training   = True,     # whether to allow `training = True` or not
    
    mha_type    = 'rel_mha',
    use_causal_attention    = True,
    
    ffn1_dim     = 1024,
    ffn1_activation  = 'swish',
    
    ffn2_dim     = 1024,
    ffn2_activation  = 'swish'
)

HParamsConformerEncoder   = HParams(
    ** {k : v for k, v in HParamsConformerSubsampler.items() if k != 'embedding_dim'},
    ** HParamsConformerLayer,
    ** _base_enc_dec_kwargs,
    max_input_length    = -1,
    unique_pos_biases   = False,
    pos_embedding_type  = 'sin',
    scale_embedding = False,
    
    speaker_embedding_dim   = -1,
    concat_mode = ConcatMode.CONCAT,
    concat_pos  = ConcatPos.AFTER_SAMPLING
)

HParamsConformerTransducer  = HParamsRNNT(
    ** HParamsConformerEncoder.get_config(add_prefix = 'encoder'),
    ** HParamsRNNTDecoder.get_config(add_prefix = 'decoder'),
    ** HParamsRNNTJointNet.get_config(add_prefix = 'joint')
)

def _get_var(_vars, i):
    if callable(_vars) and _vars.__class__.__name__ == 'function': return _vars(i)
    elif isinstance(_vars, list): return _vars[i]
    else: return _vars

class ConformerSubsampler(tf.keras.Model):
    def __init__(self, embedding_dim, n_mel_channels, name = 'feature_extractor', ** kwargs):
        super().__init__(name = name)
        
        self.hparams = HParamsConformerSubsampler.extract({
            ** kwargs, 'embedding_dim' : embedding_dim, 'n_mel_channels' : n_mel_channels
        })
        
        self.last_filters   = n_mel_channels
        self.subsampling_factor = 1
        
        self.dropout = tf.keras.layers.Dropout(
            self.hparams.subsampler_drop_rate
        ) if self.hparams.subsampler_drop_rate > 0. else None
        
        self.conv_layers    = []
        self.act_layers     = []
        self.paddings       = []
        self._strides       = []
        
        for i in range(self.hparams.n_conv):
            conv_config = {
                k : _get_var(self.hparams['subsampler_{}'.format(k)], i)
                for k in _conv_kwargs.keys()
            }
            if conv_config['filters'] == -1 or i == self.hparams.n_conv - 1:
                conv_config['filters']  = self.hparams.embedding_dim
                self.last_filters       = conv_config['filters']
            
            self.subsampling_factor *= conv_config['strides']

            self.conv_layers.append(tf.keras.layers.Conv2D(
                ** conv_config, padding = 'valid', name = 'conv_{}'.format(i)
            ))
            self.act_layers.append(get_activation(
                _get_var(self.hparams.subsampler_activation, i)
            ))
            self.paddings.append(conv_config['kernel_size'] // 2)
            self._strides.append(conv_config['strides'])
        
        self.out_layer = tf.keras.layers.Dense(self.hparams.embedding_dim)
        
        self.subsampling_factor = tf.constant(self.subsampling_factor, dtype = tf.int32)
        self.feat_length = tf.constant(
            (self.last_filters * n_mel_channels) // self.subsampling_factor, dtype = tf.int32
        )
    
    @timer(name = 'sampler call')
    def call(self, inputs, input_length = None, training = False, ** kwargs):
        features = tf.expand_dims(inputs, axis = -1)
        
        for conv, act, pad in zip(self.conv_layers, self.act_layers, self.paddings):
            if pad > 0:
                features = tf.pad(
                    features, [(0,0), (pad, pad), (pad, pad), (0, 0)]
                )
            features = conv(features, training = training)
            if act is not None:             features = act(features)
            if self.dropout is not None:    features = self.dropout(features, training = training)
        
        output = tf.reshape(
            tf.transpose(features, [0, 1, 3, 2]),
            [tf.shape(features)[0], tf.shape(features)[1], self.feat_length]
        )
        output = self.out_layer(output)
        
        if input_length is not None:
            for stride in self._strides:
                input_length = tf.cast(tf.math.floor(input_length / stride + 1.), tf.int32)
        
        return output, input_length

    def get_config(self):
        return self.hparams.get_config()
    
    @classmethod
    def from_config(cls, config, custom_objects = None):
        return cls(** config)
    
class ConformerConvolution(tf.keras.Model):
    def __init__(self, embedding_dim, name = 'conformer_conv', ** kwargs):
        super().__init__(name = name)
        
        self.hparams = HParamsConformerConvolution.extract({
            ** kwargs, 'embedding_dim' : embedding_dim
        })
        
        self.conv1  = tf.keras.layers.Conv1D(
            embedding_dim * 2, kernel_size = 1, strides = 1, use_bias = True, name = 'conv_1'
        )
        self.conv2  = tf.keras.layers.Conv1D(
            embedding_dim,
            kernel_size = self.hparams.kernel_size,
            strides     = 1,
            use_bias    = True,
            padding     = 'same',
            groups      = embedding_dim,
            name        = 'conv_2'
        )
        
        self.norm   = None
        if self.hparams.conv_norm_type in ('batch', 'batch_norm'):
            self.norm   = tf.keras.layers.BatchNormalization(
                epsilon = self.hparams.epsilon, name = 'batch_norm'
            )
        elif self.hparams.conv_norm_type in ('layer', 'layer_norm'):
            self.norm   = tf.keras.layers.LayerNormalization(
                epsilon = self.hparams.epsilon, name = 'norm'
            )
        else:
            raise ValueError("Unknown norm type : {}".format(self.hparams.conv_norm_type))
        
        self.activation = get_activation(self.hparams.conv_activation)
        self.conv3  = tf.keras.layers.Conv1D(
            embedding_dim, kernel_size = 1, strides = 1, use_bias = True, name = 'conv_3'
        )
        
    def call(self, inputs, input_length = None, training = False, ** kwargs):
        x = glu(self.conv1(inputs, training = training))
        
        x = self.conv2(x, training = training)
        x = self.norm(x, training = training)
        
        x = self.activation(x)
        x = self.conv3(x, training = training)
        return x

    def get_config(self):
        return self.hparams.get_config()
    
    @classmethod
    def from_config(cls, config, custom_objects = None):
        return cls(** config)

class ConformerLayer(tf.keras.Model):
    def __init__(self, embedding_dim, pos_bias_u = None, pos_bias_v = None,
                 name = 'conformer_layer', ** kwargs):
        super().__init__(name = name)
        
        self.hparams    = HParamsConformerLayer.extract(kwargs)
        self.hparams    = self.hparams(
            embedding_dim   = embedding_dim,
            mha_epsilon     = self.hparams.epsilon,
            mha_attention_dim   = embedding_dim,
            mha_norm_training   = self.hparams.norm_training,
            mha_drop_rate   = 0.,
            mha_normalize   = False,
            mha_residual    = False
        )
        
        self.use_causal_attention   = self.hparams.use_causal_attention
        self.norm_training  = self.hparams.norm_training
        self.fc_factor  = tf.constant(self.hparams.fc_factor, dtype = tf.float32)
        
        self.norm_ffn1  = tf.keras.layers.LayerNormalization(
            epsilon = self.hparams.epsilon, name = 'norm_ffn1'
        )
        self.ffn1       = FeedForwardNetwork(
            self.hparams.ffn1_dim, self.hparams.ffn1_activation, embedding_dim, name = 'ffn1'
        )
        
        self.norm_conv  = tf.keras.layers.LayerNormalization(
            epsilon = self.hparams.epsilon, name = 'norm_conv'
        )
        self.conv       = ConformerConvolution(** self.hparams, name = 'conv')
        
        self.norm_attn  = tf.keras.layers.LayerNormalization(
            epsilon = self.hparams.epsilon, name = 'norm_mha'
        )
        if self.hparams.mha_type == 'rel_mha':
            self.attn   = RelPosMultiHeadAttention(
                ** self.hparams.get_config(prefix = 'mha'),
                pos_bias_u = pos_bias_u,
                pos_bias_v = pos_bias_v,
                name    = 'rel_mha'
            )
        elif self.hparams.mha_type == 'mha':
            self.attn   = MultiHeadAttention(
                ** self.hparams.get_config(prefix = 'mha'), name = 'mha'
            )
        else:
            raise ValueError('Unknown MHA type : {}'.format(self.hparams.mha_type))
        
        self.norm_ffn2  = tf.keras.layers.LayerNormalization(
            epsilon = self.hparams.epsilon, name = 'norm_ffn2'
        )
        self.ffn2       = FeedForwardNetwork(
            self.hparams.ffn2_dim, self.hparams.ffn2_activation, embedding_dim, name = 'ffn2'
        )
        
        self.dropout    = tf.keras.layers.Dropout(self.hparams.drop_rate)
        self.norm       = tf.keras.layers.LayerNormalization(
            epsilon = self.hparams.epsilon, name = 'norm_output'
        )

        
    @timer(name = 'layer call')
    def call(self,
             inputs,
             input_length   = None,
             pos_emb    = None,
             initial_state  = None,
             mask       = None,
             padding_mask   = None,
             look_ahead_mask    = None,
             training   = False,
             return_state       = False,
             return_attention   = True,
             
             ** kwargs
            ):
        if padding_mask is None:
            padding_mask = build_mask(
                inputs, False, input_length = input_length, initial_state = initial_state
            )
        if mask is None:
            mask = build_mask(
                inputs, self.use_causal_attention, input_length = input_length,
                look_ahead_mask = look_ahead_mask, padding_mask = padding_mask,
                initial_state = initial_state
            )

        residual = inputs
        
        x = self.norm_ffn1(inputs, training = training and self.norm_training)
        x = self.ffn1(x, training = training)
        residual = residual + self.dropout(x, training = training) * self.fc_factor
        
        x = self.norm_attn(residual, training = training and self.norm_training)
        attn_outputs = self.attn(
            x, x, x, pos_emb = pos_emb, mask = mask, training = training,
            return_attention = return_attention, return_state = return_state
        )
        if not isinstance(attn_outputs, tuple): attn_outputs = (attn_outputs, )
        x = attn_outputs[0]
        residual = residual + self.dropout(x, training = training)

        x = self.norm_conv(residual, training = training and self.norm_training)
        x = self.conv(x, mask = padding_mask, training = training)
        residual = residual + self.dropout(x, training = training)

        x = self.norm_ffn2(residual, training = training and self.norm_training)
        x = self.ffn2(x, training = training)
        residual = residual + self.dropout(x, training = training) * self.fc_factor
        
        output = self.norm(residual, training = training and self.norm_training)
        
        return output if len(attn_outputs) == 1 else ((output,) + attn_outputs[1:])

    def get_config(self):
        return self.hparams.get_config()
    
    @classmethod
    def from_config(cls, config, custom_objects = None):
        return cls(** config)

class ConformerEncoder(TransformerBlock):
    default_params  = HParamsConformerEncoder
    _attr_to_set    = TransformerBlock._attr_to_set + [
        'n_mel_channels', 'scale_embedding', 'speaker_embedding_dim', 'concat_pos', 'concat_mode'
    ]
    
    def __init__(self, embedding_dim, n_mel_channels, max_input_length, name = None, ** kwargs):
        super(TransformerBlock, self).__init__(name = name)
        self.hparams    = self.default_params.extract(kwargs)
        self.hparams    = self.hparams(
            embedding_dim       = embedding_dim,
            n_mel_channels      = n_mel_channels,
            max_input_length    = max_input_length
        )
        
        for config in self._attr_to_set:
            setattr(self, config, self.hparams[config])
        
        if self.speaker_embedding_dim > 0:
            self.concat_pos     = get_enum_item(self.concat_pos, ConcatPos)
            self.concat_mode    = get_enum_item(self.concat_mode, ConcatMode)
        
        self.subsampler_layer = ConformerSubsampler(** self.hparams, name = 'conv_subsampler')

        self.use_pos_emb    = False
        self.pos_bias_u, self.pos_bias_v = None, None
        if self.hparams.pos_embedding_type == 'learnable':
            pos_embed_class = FasterEmbedding
        elif self.hparams.mha_type == 'rel_mha':
            self.use_pos_emb    = True
            pos_embed_class = RelPositionalEmbedding
            if self.hparams.unique_pos_biases:
                features_dim = embedding_dim if (
                    self.speaker_embedding_dim <= 0
                    or self.concat_mode != ConcatMode.CONCAT
                    or self.concat_pos == ConcatPos.END
                ) else embedding_dim + self.speaker_embedding_dim
                
                mha_depth = features_dim // self.hparams.mha_num_heads
                self.pos_bias_u = self.add_weight(
                    shape   = [self.hparams.mha_num_heads, mha_depth],
                    name    = "pos_bias_u",
                    trainable   = True,
                    initializer = "zeros"
                )
                self.pos_bias_v = self.add_weight(
                    shape   = [self.hparams.mha_num_heads, mha_depth],
                    name    = "pos_bias_v",
                    trainable   = True,
                    initializer = "zeros"
                )
        else:
            pos_embed_class = PositionalEmbedding
        
        if self.speaker_embedding_dim > 0 and self.concat_mode == ConcatMode.CONCAT and self.concat_pos == ConcatPos.BEFORE_POS:
            embedding_dim += self.speaker_embedding_dim

        self.pos_embedding_layer    = pos_embed_class(
            embedding_dim,
            max_input_length,
            name    = 'positional_encoding'
        )
        
        if self.speaker_embedding_dim > 0 and self.concat_mode == ConcatMode.CONCAT and self.concat_pos == ConcatPos.BEFORE_SCALING:
            embedding_dim += self.speaker_embedding_dim

        self.embedding_factor = tf.math.sqrt(
            float(embedding_dim) if self.hparams.scale_embedding else 1.
        )
        
        if self.speaker_embedding_dim > 0 and self.concat_mode == ConcatMode.CONCAT and self.concat_pos == ConcatPos.AFTER_SCALING:
            embedding_dim += self.speaker_embedding_dim

        self._layers = [
            ConformerLayer(
                ** self.hparams(embedding_dim = embedding_dim),
                pos_bias_u  = self.pos_bias_u,
                pos_bias_v  = self.pos_bias_v,
                name    = 'layer_{}'.format(i+1)
            )
            for i in range(self.hparams.num_layers)
        ]
        
        self.resize_spk_embed_layer = None
        if self.speaker_embedding_dim > 0 and self.speaker_embedding_dim != self.embedding_dim and self.concat_mode != ConcatMode.CONCAT:
            self.resize_spk_embed_layer = tf.keras.layers.Dense(
                self.embedding_dim, name = 'resize_speaker_embedding_layer'
            )
    
    @property
    def dummy_inputs(self):
        batch_size, seq_len = 2, 128
        
        inputs  = [
            tf.random.normal((batch_size, seq_len, self.n_mel_channels)),
            tf.fill([batch_size], seq_len)
        ]
        if self.speaker_embedding_dim > 0:
            inputs.append(tf.random.normal((batch_size, self.speaker_embedding_dim)))
        
        return inputs
    
    def _concat(self, features, speaker_embedding):
        if self.speaker_embedding_dim <= 0: return features
        assert speaker_embedding is not None, "You must provide `speaker_embedding` !"
        
        if self.resize_spk_embed_layer is not None:
            speaker_embedding = self.resize_spk_embed_layer(speaker_embedding)
        
        if self.concat_mode == ConcatMode.CONCAT:
            sequence_length = tf.shape(features)[1]

            speaker_embedding = tf.expand_dims(speaker_embedding, 1)
            speaker_embedding = tf.tile(speaker_embedding, [1, sequence_length, 1])

            return tf.concat([features, speaker_embedding], axis = -1)
        elif self.concat_mode == ConcatMode.ADD:
            return features + speaker_embedding
        elif self.concat_mode == ConcatMode.SUB:
            return features - speaker_embedding
        elif self.concat_mode == ConcatMode.MUL:
            return features * speaker_embedding
        elif self.concat_mode == ConcatMode.DIV:
            return features / speaker_embedding
        else:
            return features

    @timer
    def call(self,
             inputs,
             input_length   = None,
             speaker_embedding  = None,
             initial_state  = None,
             mask       = None,
             padding_mask   = None,
             look_ahead_mask    = None,
             training   = False,
             return_attention   = None,
             ** kwargs
            ):
        """
            Arguments :
                - inputs    : encoder inputs with shape [batch_size, seq_len, embedding_dim], embedded inputs
                - input_length  : the inputs' lengths with shape (batch_size, 1)
                - speaker_embedding : speaker embedding if used (if self.speaker_embedding_dim > 0) with shape (batch_size, self.speaker_embedding_dim)
                - mask      : attention mask (padding mask based in inputs)
                - training  : whether it is training / inference phase
                - return_attention  : whether to return attention weights or not
                - return_states     : whether to return intermediate representation or not
            Return : output if not return_attention else [output, attention]
                - output    : the layer output with same shape as input
                - attention_weights : dict self-attention weights for each head of the MHA of each layer
        """
        if isinstance(inputs, (list, tuple)):
            if len(inputs) == 3: speaker_embedding = inputs[2]
            inputs, input_length = inputs[:2]
        
        features, input_length = self.subsampler_layer(
            inputs, input_length = input_length, training = training, mask = mask
        )

        if self.concat_pos == ConcatPos.BEFORE_POS:
            features = self._concat(features, speaker_embedding)
        
        pos_emb = None
        if self.use_pos_emb:
            features, pos_emb = self.pos_embedding_layer(features)
        else:
            features = self.pos_embedding_layer(features)

        if self.concat_pos == ConcatPos.BEFORE_SCALING:
            features = self._concat(features, speaker_embedding)

        if self.scale_embedding:
            features = features * self.embedding_factor
        
        if self.concat_pos == ConcatPos.AFTER_SCALING:
            features = self._concat(features, speaker_embedding)
        
        padding_mask = build_mask(
            features, False, input_length = input_length, initial_state = initial_state
        )
        mask = build_mask(
            features, self.use_causal_attention, input_length = input_length,
            look_ahead_mask = look_ahead_mask, padding_mask = padding_mask,
            initial_state = initial_state[0] if initial_state is not None else None
        )
        
        output = super().call(
            features,
            pos_emb = pos_emb,
            input_length    = input_length,
            training    = training,
            mask    = mask,
            padding_mask    = padding_mask,
            return_attention    = False,
            ** kwargs
        )
        if isinstance(output, (list, tuple)): output = output[0]
        
        if self.concat_pos == ConcatPos.END:
            features = self._concat(output, speaker_embedding)

        return (output, input_length)
    
    def transfer_weights(self, pretrained):
        from models.weights_converter import get_pt_variables, partial_transfer_learning
        
        weights = get_pt_variables(pretrained)
        # push 'linear_out' at the end of the subsampler's weights
        weights = weights[2:6] + weights[:2] + weights[6:]
        # re-transpose the pos_biasses
        weights = [w if self.hparams.mha_num_heads not in w.shape else w.T for w in weights]
        
        if self.hparams.conv_norm_type in ('batch', 'batch_norm'):
            offset, layer_offset, n_layer = 6, 14, 39
            indexes = []
            for i in range(self.hparams.num_layers):
                indexes.extend([
                    offset + layer_offset + n_layer * i,
                    offset + layer_offset + 1 + n_layer * i
                ])

            weights = weights + list(reversed([weights.pop(i) for i in reversed(indexes)]))
        
        partial_transfer_learning(self, weights)

    @classmethod
    def from_pretrained(cls,
                        pretrained_name = 'stt_en_conformer_transducer_medium',
                        pretrained      = None,
                        ** kwargs
                       ):
        if pretrained is None:
            pretrained = nemo_conformer(pretrained_name)
        
        cfg = pretrained.cfg.encoder
        
        config = HParamsConformerEncoder(
            embedding_dim   = cfg.d_model,
            n_mel_channels  = cfg.feat_in,
            max_input_length    = cfg.pos_emb_max_len,
            unique_pos_biases   = cfg.untie_biases,
            
            
            epsilon = 1e-5,
            scale_embedding = cfg.xscaling,

            n_conv  = int(math.log(cfg.subsampling_factor, 2)),
            subsampler_filters  = cfg.subsampling_conv_channels,
            
            num_layers  = cfg.n_layers,
            
            conv_norm_type      = cfg.conv_norm_type,
            conv_kernel_size    = cfg.conv_kernel_size,
            
            ffn1_dim    = cfg.d_model * cfg.ff_expansion_factor,
            ffn1_activation = 'swish',
            ffn2_dim    = cfg.d_model * cfg.ff_expansion_factor,
            ffn2_activation = 'swish',
            
            mha_type   = 'rel_mha' if cfg.self_attention_model == 'rel_pos' else 'mha',
            mha_num_heads   = cfg.n_heads
        )
        
        instance = cls(** config(** kwargs))
        instance._build()

        instance.transfer_weights(pretrained.encoder)
        
        return instance

class ConformerTransducer(RNNT):
    default_hparams = HParamsConformerTransducer

    encoder_class   = ConformerEncoder
    
    def transfer_weights(self, pretrained):
        from models.weights_converter import get_pt_variables, partial_transfer_learning
        
        self.encoder.transfer_weights(pretrained.encoder)
        
        dec_weights = get_pt_variables(pretrained.decoder)
        dec_weights[0] = dec_weights[0].T
        
        joint_weights   = get_pt_variables(pretrained.joint)
        
        partial_transfer_learning(self.decoder, dec_weights)
        partial_transfer_learning(self.joint_net, joint_weights)
        

    @classmethod
    def from_pretrained(cls,
                        pretrained_name = 'stt_en_conformer_transducer_medium',
                        pretrained      = None,
                        ** kwargs
                       ):
        if pretrained is None:
            pretrained = nemo_conformer(pretrained_name)
        
        cfg = pretrained.cfg.encoder
        
        sos_id = pretrained.decoding.tokenizer.bos_id
        pad_id = pretrained.decoding.tokenizer.pad_id
        config = HParamsConformerTransducer(
            vocab_size  = pretrained.cfg.decoder.vocab_size + 1,
            sos_token   = sos_id if sos_id != -1 else pretrained.cfg.decoder.vocab_size,
            pad_token   = pad_id if pad_id != -1 else pretrained.cfg.decoder.vocab_size,
            
            encoder_embedding_dim   = cfg.d_model,
            encoder_n_mel_channels  = cfg.feat_in,
            encoder_max_input_length    = cfg.pos_emb_max_len,
            encoder_unique_pos_biases   = cfg.untie_biases,
            
            encoder_epsilon = 1e-5,
            encoder_scale_embedding = cfg.xscaling,

            encoder_n_conv  = int(math.log(cfg.subsampling_factor, 2)),
            encoder_subsampler_filters  = cfg.subsampling_conv_channels,
            
            encoder_num_layers  = cfg.n_layers,
            
            encoder_conv_norm_type      = cfg.conv_norm_type,
            encoder_conv_kernel_size    = cfg.conv_kernel_size,
            
            encoder_ffn1_dim    = cfg.d_model * cfg.ff_expansion_factor,
            encoder_ffn1_activation = 'swish',
            encoder_ffn2_dim    = cfg.d_model * cfg.ff_expansion_factor,
            encoder_ffn2_activation = 'swish',
            
            encoder_mha_type   = 'rel_mha' if cfg.self_attention_model == 'rel_pos' else 'mha',
            encoder_mha_num_heads   = cfg.n_heads,
            
            decoder_embedding_dim   = pretrained.cfg.decoder.prednet.pred_hidden,
            
            joint_embedding_dim = pretrained.cfg.joint.jointnet.joint_hidden,
            joint_activation    = pretrained.cfg.joint.jointnet.activation
        )
        
        instance = cls(** config(** kwargs))
        instance._build()

        instance.transfer_weights(pretrained)
        
        return instance

def nemo_conformer(model_name):
    import nemo.collections.asr as nemo_asr
    
    return nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name)

_conformer_classes   = {
    'ConformerSubsampler'   : ConformerSubsampler,
    'ConformerConvolution'  : ConformerConvolution,
    'ConformerLayer'    : ConformerLayer,
    'ConformerEncoder'  : ConformerEncoder,
    'ConformerTransducer'   : ConformerTransducer
}
        
custom_functions    = {
    ** _conformer_classes
}

custom_objects  = {
    ** _conformer_classes,
    'PositionalEmbedding'   : PositionalEmbedding,
    'RelPositionalEmbedding'    : RelPositionalEmbedding,
    'RelPosMultiHeadAttention'  : RelPosMultiHeadAttention,
    'MultiHeadAttention'    : MultiHeadAttention,
    'FeedForwardNetwork'    : FeedForwardNetwork
}