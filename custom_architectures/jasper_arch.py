
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

import os
import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras.mixed_precision import experimental as mixed_precision

from custom_architectures.current_blocks import _get_var, Conv1DBN
from custom_layers import get_activation, LogSoftmax

_pretrained_weights_file = os.path.join(
    'pretrained_models', 'pretrained_weights', 'pretrained_jasper.h5'
)
    
def JasperBlock(inputs,
                filters,
                repeat      = 3,
                kernel_size     = 11,
                strides     = 1,
                bias        = False,
                dilation    = 1,
                padding     = 'same',
                dropout     = 0.2,
                activation  = None, 
                residual    = True,
                residual_inputs = [],
                use_conv_mask   = False,
                name        = None
               ):
    if activation is None: activation = tf.keras.layers.ThresholdedReLU(theta = 20.)
    name = '' if name is None else name + '_'

    x = inputs

    for i in range(repeat):
        activation_i    = activation if i < repeat - 1 else None
        dropout_i       = dropout if i < repeat-1 else 0.

        x = Conv1DBN(
            x,
            filters         = filters,
            kernel_size     = kernel_size,
            strides         = strides,
            dilation_rate   = dilation,
            padding         = padding,
            use_bias        = bias,
            bnorm           = 'after',
            activation      = activation_i,
            drop_rate       = dropout_i,
            residual        = False,
            name            = '{}conv_{}'.format(name, i+1)
        )
        
    if residual:
        if len(residual_inputs) == 0: residual_inputs = [inputs]
        
        residual_outputs = []
        for i, res_input in enumerate(residual_inputs):
            model = Conv1DBN(
                tf.keras.Sequential(name = '{}_model_residual_{}'.format(name, i+1)),
                filters         = filters,
                kernel_size     = 1,
                strides         = 1,
                dilation_rate   = 1,
                padding         = padding,
                use_bias        = bias,
                bnorm           = 'after',
                activation      = None,
                drop_rate       = 0.,
                residual        = False,
                name            = '{}residual_{}'.format(name, i+1)
            )
            out = model(res_input)
            
            residual_outputs.append(out)
        
        x = tf.keras.layers.Add()([x] + residual_outputs)
        
    if activation is not None:
        x = get_activation('relu')(x)
    
    if dropout > 0.: x = tf.keras.layers.Dropout(dropout)(x)
        
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
           dropout       = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3, 0.4, 0.4],
           residual      = [False, True, True, True, True, True, True, True, True, True, True, False, False],
           residual_dense    = [False, True, True, True, True, True, True, True, True, True, True, False, False],
           last_activation   = 'log_softmax',
           use_mixed_precision  = False,
           pretrained           = False,
           pretrained_file      = _pretrained_weights_file,
           name  = 'Jasper'
          ):
    if use_mixed_precision:
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_policy(policy)

    inputs = tf.keras.layers.Input(shape = input_shape, name = 'mel_input')
    
    x = inputs
    res = []
    for i in range(n_block):
        if _get_var(residual_dense, i): res.append(x)
        x = JasperBlock(
            x,
            repeat      = _get_var(repeat, i),
            filters     = _get_var(filters, i),
            kernel_size = _get_var(kernel_size, i),
            strides     = _get_var(strides, i),
            dilation    = _get_var(dilation, i),
            activation  = _get_var(activation, i),
            dropout     = _get_var(dropout, i),
            residual    = _get_var(residual, i),
            residual_inputs = res,
            name        = "block_{}".format(i+1)
        )

    out = tf.keras.layers.Conv1D(
        filters = vocab_size, kernel_size = 1, activation = None, name = 'output_layer'
    )(x)
    out = get_activation(last_activation)(out)
    
    model = tf.keras.Model(inputs, out, name = name)
    
    if pretrained:
        model.load_weights(pretrained_file)
    
    return model

custom_functions    = {
    'Jasper'    : Jasper
}

custom_objects  = {
    "log_softmax"   : LogSoftmax,
    "LogSoftmax"    : LogSoftmax,
}