# Copyright (C) 2022-now yui-mhcp project author. All rights reserved.
# Licenced under a modified Affero GPL v3 Licence (the "Licence").
# you may not use this file except in compliance with the License.
# See the "LICENCE" file at the root of the directory for the licence information.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import keras

from keras import layers

from .current_blocks import _get_var, Conv1DBN, MaskedConv1DBN
from custom_layers import get_activation, LogSoftmax, MaskedConv1D

_pretrained_weights_file = os.path.join(
    'pretrained_models', 'pretrained_weights', 'pretrained_jasper.h5'
)

def jasper_block(inputs,
                 filters,
                 use_mask,
                 repeat      = 3,
                 kernel_size     = 11,
                 strides     = 1,
                 use_bias    = False,
                 dilation    = 1,
                 padding     = 'same',
                    
                 drop_rate   = 0.2,

                 activation  = None,
                
                 residual    = True,
                 residual_inputs = [],

                 pretrained  = False,
                 name        = None
                ):
    conv_fn = Conv1DBN if not use_mask else MaskedConv1DBN
    
    if activation is None: activation = layers.ThresholdedReLU(theta = 20.)
    name = '' if name is None else name + '_'

    x = inputs
    for i in range(repeat):
        activation_i    = activation if i < repeat - 1 else None
        drop_rate_i     = drop_rate if i < repeat-1 else 0.

        x = conv_fn(
            x,
            filters         = filters,
            kernel_size     = kernel_size,
            strides         = strides,
            dilation_rate   = dilation,
            padding         = padding,
            use_bias        = use_bias,
            bnorm           = 'after',
            activation      = activation_i,
            drop_rate       = drop_rate_i,
            residual        = False,
            use_mask    = use_mask,
            use_manual_padding  = use_mask,
            bn_name         = '{}norm_{}'.format(name, i+1),
            name            = '{}conv_{}'.format(name, i+1)
        )
    
    if residual:
        if len(residual_inputs) == 0: residual_inputs = [inputs]
        
        residual_outputs = []
        for i, res_input in enumerate(residual_inputs):
            if pretrained:
                residual_outputs.append(conv_fn(
                    keras.Sequential(name = '{}_model_residual_{}'.format(name, i+1)),
                    filters         = filters,
                    kernel_size     = 1,
                    strides         = 1,
                    dilation_rate   = 1,
                    padding         = padding,
                    use_bias        = use_bias,
                    bnorm           = 'after',
                    activation      = None,
                    drop_rate       = 0.,
                    residual        = False,
                    bn_name         = '{}residual_norm_{}'.format(name, i+1),
                    name            = '{}residual_{}'.format(name, i+1)
                )(res_input))
            else:
                residual_outputs.append(conv_fn(
                    res_input,
                    filters         = filters,
                    kernel_size     = 1,
                    strides         = 1,
                    dilation_rate   = 1,
                    padding         = padding,
                    use_bias        = use_bias,
                    bnorm           = 'after',
                    activation      = None,
                    drop_rate       = 0.,
                    residual        = False,
                    bn_name         = '{}residual_norm_{}'.format(name, i+1),
                    name            = '{}residual_{}'.format(name, i+1)
                ))
        
        x = layers.Add()([x] + residual_outputs)
    
    if activation is not None:
        x = get_activation('relu')(x)
    
    if drop_rate > 0.: x = layers.Dropout(drop_rate)(x)
        
        
    return x
        
def Jasper(input_shape,
           vocab_size,
           n_block       = 13,
           filters       = [256, 256, 256, 384, 384, 512, 512, 640, 640, 768, 768, 896, 1024],
           repeat        = [1, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 1, 1],
           kernel_size   = [11, 11, 11, 13, 13, 17, 17, 21, 21, 25, 25, 29, 1] ,
           strides       = [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ,
           dilation      = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1] ,
           activation    = 'relu',
           drop_rate     = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3, 0.4, 0.4],
           residual      = [False, True, True, True, True, True, True, True, True, True, True, False, False],
           residual_dense    = [False, True, True, True, True, True, True, True, True, True, True, False, False],
           last_activation   = 'log_softmax',
           
           use_mixed_precision  = False,
           pretrained           = False,
           pretrained_file      = _pretrained_weights_file,
           
           pad_value    = None,
           name  = 'Jasper',
           
           ** kwargs
          ):
    use_mask    = pad_value is not None
    last_conv_fn    = MaskedConv1D if use_mask else layers.Conv1D
    if use_mixed_precision:
        policy = keras.mixed_precision.set_global_policy('mixed_float16')
    
    inputs = layers.Input(shape = input_shape, name = 'mel_input')
    
    x = inputs if not use_mask else keras.layers.Masking(mask_value = pad_value)(inputs)
    res = []
    for i in range(n_block):
        if _get_var(residual_dense, i): res.append(x)
        x = JasperBlock(
            x,
            use_mask    = use_mask,
            repeat      = _get_var(repeat, i),
            filters     = _get_var(filters, i),
            kernel_size = _get_var(kernel_size, i),
            strides     = _get_var(strides, i),
            dilation    = _get_var(dilation, i),
            activation  = _get_var(activation, i),
            drop_rate   = _get_var(drop_rate, i),
            residual    = _get_var(residual, i),
            residual_inputs = res,
            pretrained  = pretrained,
            name        = "block_{}".format(i+1)
        )

    out = last_conv_fn(
        filters = vocab_size, kernel_size = 1, padding = 'same', activation = None, name = 'output_layer'
    )(x)
    out = get_activation(last_activation)(out)
    
    model = keras.Model(inputs, out, name = name)
    
    if pretrained: model.load_weights(pretrained_file)
    
    return model
