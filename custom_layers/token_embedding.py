
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

import tensorflow as tf

class TokenEmbedding(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, vocab_size, max_len, ** kwargs):
        super().__init__(** kwargs)
        
        self.vocab_size = vocab_size
        self.max_len    = max_len
        self.embedding_dim  = embedding_dim
        
        self.embedding_layer    = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.pos_embedding_layer    = tf.keras.layers.Embedding(max_len, embedding_dim)
        
        
    def call(self, tokens, training = False):
        embedded = self.embedding_layer(tokens, training = training)
        pos_embedded    = self.pos_embedding_layer(tf.range(0, tf.shape(tokens)[1]), training = training)
        return embedded + pos_embedded
    
    def get_config(self):
        config = super().get_config()
        config['vocab_size'] = self.vocab_size
        config['max_len']    = self.max_len
        config['embedding_dim']  = self.embedding_dim
        return config
        