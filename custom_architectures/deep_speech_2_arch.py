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
import logging
import numpy as np

from keras import layers

logger  = logging.getLogger(__name__)

def DeepSpeech2(input_shape,
                vocab_size,
                is_mixed_precision  = False,
                rnn_units       = 800, 
                random_state    = 1,
                pretrained      = False
               ):
    if is_mixed_precision:
        policy = keras.mixed_precision.set_global_policy('mixed_float16')

    # Define input tensor [batch, time, features]
    input_tensor = layers.Input(shape = input_shape, name = 'input')

    # Add 4th dimension [batch, time, frequency, channel]
    x = K.expand_dims(input_tensor, axis = -1)

    x = layers.Conv2D(
        filters     = 32,
        kernel_size = [11, 41],
        strides     = [2, 2],
        padding     = 'same',
        use_bias    = False,
        name        = 'conv_1'
    )(x)
    x = layers.BatchNormalization(name = 'conv_1_bn')(x)
    x = layers.ReLU(name = 'conv_1_relu')(x)

    x = layers.Conv2D(
        filters     = 32,
        kernel_size = [11, 21],
        strides     = [1, 2],
        padding     = 'same',
        use_bias    = False,
        name        = 'conv_2'
    )(x)
    x = layers.BatchNormalization(name = 'conv_2_bn')(x)
    x = layers.ReLU(name = 'conv_2_relu')(x)
    # We need to squeeze to 3D tensor. Thanks to the stride in frequency
    # domain, we reduce the number of features four times for each channel.
    seq_len = input_shape[0] // 2 if input_shape[0] is not None else -1
    reshaped = (None, seq_len, input_shape[1] // 4 * 32)

    x = layers.Reshape((seq_len, input_shape[1] // 4 * 32))(x)

    for i in [1, 2, 3, 4, 5]:
        x = layers.Bidirectional(layers.GRU(
            units       = rnn_units,
            activation  = 'tanh',
            recurrent_activation    = 'sigmoid',
            use_bias    = True,
            return_sequences    = True,
            reset_after = True,
            name        = f'gru_{i}'
        ), name = 'bidirectional_{}'.format(i), merge_mode = 'concat')(x)
        if i > 5: x = layers.Dropout(0.5)(x)

    # Return at each time step logits along characters. Then CTC
    # computation is more stable, in contrast to the softmax.
    x = layers.TimeDistributed(
        layers.Dense(rnn_units * 2), name = 'dense_1'
    )(x)
    x = layers.ReLU(name = 'dense_1_relu')(x)
    x = layers.Dropout(0.5)(x)
    output_tensor = layers.TimeDistributed(
        layers.Dense(vocab_size, activation = 'softmax'), name = 'output_layer'
    )(x)

    model = keras.Model(input_tensor, output_tensor, name = 'DeepSpeech2')
    
    if pretrained:
        model.load_weights(get_pretrained_weights())
    
    return model

def get_pretrained_weights(lang = 'en', name = 'deepspeech2', version = 0.1):
    bucket = 'automatic-speech-recognition'
    file_name = f'{lang}-{name}-weights-{version}.h5'
    remote_path = file_name
    local_path = os.path.join("pretrained_models", "pretrained_weights", file_name)
    maybe_download_from_bucket(bucket, remote_path, local_path)

    return local_path

def download_from_bucket(bucket_name: str, remote_path: str, local_path: str):
    """ Download the file from the public bucket. """
    from google.cloud import storage
    client = storage.Client.create_anonymous_client()
    bucket = client.bucket(bucket_name)
    blob = storage.Blob(remote_path, bucket)
    blob.download_to_filename(local_path, client=client)

def maybe_download_from_bucket(bucket_name: str, remote_path: str, local_path: str):
    """ Download file from the bucket if it does not exist. """
    if os.path.isfile(local_path):
        return
    directory = os.path.dirname(local_path)
    os.makedirs(directory, exist_ok=True)
    logger.info('Downloading file from the bucket...')
    download_from_bucket(bucket_name, remote_path, local_path)

custom_functions    = {
    'DeepSpeech2'   : DeepSpeech2,
    'deep_speech_2' : DeepSpeech2
}