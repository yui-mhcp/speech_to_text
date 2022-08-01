
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

import numpy as np
import tensorflow as tf

class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, max_len, ** kwargs):
        super().__init__(** kwargs)
        self.max_len    = max_len
        self.embedding_dim  = embedding_dim
        
        self.pos_encoding = self.get_positional_encoding()
        
    def get_angles(self, pos, i):
        angle_rates = 1. / tf.pow(10000, (2 * (i // 2)) / tf.cast(self.embedding_dim, tf.float32))

        return pos * angle_rates
        
    def get_positional_encoding(self):
        angle_rads = self.get_angles(
            tf.range(self.max_len, dtype = tf.float32)[:, tf.newaxis],
            tf.range(self.embedding_dim, dtype = tf.float32)[tf.newaxis, :],
        )

        angle_rads = angle_rads.numpy()
        # Applique le sinus sur les indices pairs (2 i). 
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        # Applique le sinus sur les indices pairs (2 i + 1). 
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype = tf.float32)
    
    def call(self, inputs, training = False):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]
    
    def get_config(self):
        config = super().get_config()
        config['max_len']    = self.max_len
        config['embedding_dim']  = self.embedding_dim
        return config

class RelPositionalEmbedding(PositionalEmbedding):
    def get_positional_encoding(self):
        positions = tf.expand_dims(
            tf.range(self.max_len - 1, - self.max_len, -1, dtype = tf.float32), axis = -1
        )
        
        div_term = tf.exp(
            tf.range(0, self.embedding_dim, 2, dtype = tf.float32) * -(tf.math.log(10000.) / self.embedding_dim)
        )

        pos_encoding = np.zeros((len(positions), self.embedding_dim))
        
        # Applique le sinus sur les indices pairs (2 i). 
        pos_encoding[:, 0::2] = np.sin(positions * div_term)

        # Applique le sinus sur les indices pairs (2 i + 1). 
        pos_encoding[:, 1::2] = np.cos(positions * div_term)

        pos_encoding = pos_encoding[np.newaxis, ...]
        
        self.center_pos = tf.constant(pos_encoding.shape[1] // 2 + 1, dtype = tf.int32)
        
        return tf.cast(pos_encoding, dtype = tf.float32)

    def call(self, inputs, training = False):
        start_pos   = self.center_pos - tf.shape(inputs)[1]
        end_pos     = self.center_pos + tf.shape(inputs)[1] - 1
        
        return inputs, self.pos_encoding[:, start_pos : end_pos]
