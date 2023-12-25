
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

class TextLoss(tf.keras.losses.Loss):
    def __init__(self,
                 pad_value  = 0,
                 eos_value  = -1,
                 from_logits    = False,
                 warmup_tokens  = 0,
                 name = 'TextLoss',
                 ** kwargs
                ):
        kwargs['reduction'] = tf.keras.losses.Reduction.NONE
        super().__init__(name = name, ** kwargs)
        
        self.pad_value  = pad_value
        self.eos_value  = eos_value
        self.pad_is_eos = pad_value == eos_value
        self.from_logits    = from_logits
        self.warmup_tokens  = warmup_tokens
    
    def build_warmup_mask(self, length, dtype):
        warmups = tf.cast(tf.range(1, self.warmup_tokens + 1), dtype)
        warmups = warmups / tf.cast(self.warmup_tokens + 1, dtype)
        
        return tf.concat([
            warmups, tf.ones((tf.maximum(0, length - self.warmup_tokens), ), dtype = dtype)
        ], axis = -1)[tf.newaxis, : length]
    
    def compute_padding_mask(self, y_true, target_length, dtype = tf.float32):
        return tf.sequence_mask(
            target_length, maxlen = tf.shape(y_true)[1], dtype = dtype
        )

    def call(self, y_true, y_pred, sample_weight = None, padding_mask = None):
        if not isinstance(y_pred, tf.Tensor): y_pred = y_pred[0]
        y_pred._keras_mask = None
        
        if isinstance(y_true, tuple) and len(y_true) != 2: y_true = y_true[0]
        if not isinstance(y_true, (list, tuple)):
            target_length = tf.reduce_sum(tf.cast(
                tf.math.not_equal(y_true, self.pad_value), tf.int32
            ), axis = -1)
            if self.pad_is_eos: target_length += 1
        else:
            y_true, target_length = y_true

        if len(tf.shape(y_true)) == 3:
            y_true, target_length = y_true[:, 0], target_length[:, 0]
        
        y_pred  = y_pred[:, - tf.shape(y_true)[1] :, :]
        loss    = tf.keras.losses.sparse_categorical_crossentropy(
            y_true, y_pred, from_logits = self.from_logits
        )

        #if padding_mask is None:
        padding_mask = self.compute_padding_mask(y_true, target_length, loss.dtype)
        
        loss = loss * padding_mask
        
        if self.warmup_tokens > 0:
            loss = loss * self.build_warmup_mask(tf.shape(loss)[1], loss.dtype)
        
        return tf.reduce_sum(loss, axis = -1) / tf.maximum(tf.cast(target_length, loss.dtype), 1e-6)
    
    def get_config(self):
        return {
            ** super().get_config(),
            'warmup_tokens' : self.warmup_tokens,
            'from_logits'   : self.from_logits,
            'eos_value' : self.eos_value,
            'pad_value' : self.pad_value
        }
