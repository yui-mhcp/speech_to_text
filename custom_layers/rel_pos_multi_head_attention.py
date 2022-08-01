
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

from custom_layers.multi_head_attention import MultiHeadAttention


class RelPosMultiHeadAttention(MultiHeadAttention):
    def __init__(self, pos_bias_u = None, pos_bias_v = None, ** kwargs):
        super().__init__(** kwargs)
        
        self.linear_pos = tf.keras.layers.Dense(
            self.attention_dim, use_bias = False, name = 'rel_pos_layer'
        )
        
        self.pos_bias_u = pos_bias_u
        self.pos_bias_v = pos_bias_v
        if pos_bias_u is None or pos_bias_v is None:
            self.pos_bias_u = self.add_weight(
                shape   = [self.num_heads, self.depth],
                name    = "pos_bias_u",
                trainable   = True,
                initializer = "zeros"
            )
            self.pos_bias_v = self.add_weight(
                shape   = [self.num_heads, self.depth],
                name    = "pos_bias_v",
                trainable   = True,
                initializer = "zeros"
            )

    def rel_shift(self, x):
        batch_size = tf.shape(x)[0]
        q_len = tf.shape(x)[2]
        k_len = tf.shape(x)[3]
        
        x = tf.pad(x, [(0, 0), (0, 0), (0, 0), (1, 0)])
        x = tf.reshape(x, [batch_size, self.num_heads, -1, q_len])
        return tf.reshape(x[:, :, 1:], [batch_size, self.num_heads, q_len, k_len])

    def call(self,
             query,
             key,
             value,
             pos_emb,
             mask       = None,
             training   = False,
             initial_state  = None,
             return_attention   = True,
             return_state   = False,
             normalize_kv   = True,
             
             ** kwargs
            ):
        q, k, v = self.process_qkv(
            query, key, value, training, normalize_kv = normalize_kv, initial_state = initial_state
        )
        q = tf.transpose(q, [0, 2, 1, 3])
        
        p = self.linear_pos(pos_emb)
        p = self.split_heads(p, 1)
        
        q_bias_u = tf.transpose(q + self.pos_bias_u, [0, 2, 1, 3])
        q_bias_v = tf.transpose(q + self.pos_bias_v, [0, 2, 1, 3])
        
        matrix_ac = tf.matmul(q_bias_u, k, transpose_b = True)
        
        matrix_bd = tf.matmul(q_bias_v, p, transpose_b = True)
        matrix_bd = self.rel_shift(matrix_bd)
        matrix_bd = matrix_bd[:, :, :, : tf.shape(matrix_ac)[-1]]
        
        scores  = (matrix_ac + matrix_bd) / self.sqrt_depth
        
        # scaled_attention shape == (atch, num_heads, seq_len_q, depth)
        # attention_weights shape == (batch, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = self.compute_attention(
            scores, v, mask = mask, training = training
        )

        output = self.merge_heads(scaled_attention, tf.shape(query)[0])
        
        if self.output_layer is not None:
            output = self.output_layer(output)
            
            if self.dropout is not None:    output = self.dropout(output, training = training)
            if self.residual:       output = output + query
            if self.norm_layer is not None:
                output = self.norm_layer(output, training = training and self.norm_training)
        
        output = (output, )
        if return_state:        output = output + ((k, v), )
        if return_attention:    output = output + (attention_weights, )
        return output[0] if len(output) == 1 else output
    