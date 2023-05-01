

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

import importlib
import tensorflow as tf

from loggers import timer
from hparams import HParams
from custom_layers import get_activation, FasterEmbedding
from custom_architectures.transducer_generation_utils import transducer_infer

HParamsRNNTJointNet = HParams(
    vocab_size  = -1,
    embedding_dim = 640,
    activation  = 'relu',
    drop_rate   = 0.1
)

HParamsRNNTDecoder = HParams(
    vocab_size  = -1,
    embedding_dim   = 640,
    activation  = None,
    drop_rate   = 0.1
)

HParamsRNNT = HParams(
    vocab_size  = -1,
    sos_token   = -1,
    pad_token   = -1
)

class RNNTLSTMDecoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, name = 'rnnt_decoder', ** kwargs):
        super().__init__(name = name)
        
        self.hparams = HParamsRNNTDecoder.extract({
            ** kwargs, 'vocab_size' : vocab_size, 'embedding_dim' : embedding_dim
        })
        
        self.embedding_layer    = FasterEmbedding(
            vocab_size, embedding_dim, name = 'embedding_layer'
        )
        
        self.rnn_layer  = tf.keras.layers.LSTM(
            embedding_dim, return_sequences = True, return_state = True, name = 'rnn_layer'
        )
        
        self.activation = get_activation(self.hparams.activation)
        self.dropout    = tf.keras.layers.Dropout(self.hparams.drop_rate) if self.hparams.drop_rate > 0. else None
    
    @property
    def vocab_size(self):
        return self.hparams.vocab_size
    
    @property
    def state_signature(self):
        return [
            tf.TensorSpec(shape = (None, self.hparams.embedding_dim), dtype = tf.float32),
            tf.TensorSpec(shape = (None, self.hparams.embedding_dim), dtype = tf.float32)
        ]
    
    def change_vocabulary(self, new_vocab, ** kwargs):
        self.hparams.vocab_size = len(new_vocab)
        self.embedding_layer.change_vocabulary(new_vocab, ** kwargs)
    
    def get_initial_state(self, inputs):
        return self.rnn_layer.get_initial_state(inputs)
    
    @timer(name = 'decoder call')
    def call(self, inputs, input_length, training = False, initial_state = None, ** kwargs):
        embed = self.embedding_layer(inputs)
        
        output, h, c = self.rnn_layer(embed, training = training, initial_state = initial_state)
        if self.activation is not None: output = self.activation(output)
        if self.dropout is not None: output = self.dropout(output, training = training)
        
        return output, [h, c]

    def get_config(self):
        return self.hparams.get_config()
    
    @classmethod
    def from_config(cls, config, custom_objects = None):
        return cls(** config)

class RNNTJointNet(tf.keras.Model):
    def __init__(self, vocab_size, name = 'joint_net', ** kwargs):
        super().__init__(name = name)
        
        self.hparams = HParamsRNNTJointNet.extract({
            ** kwargs, 'vocab_size' : vocab_size
        })
        
        self.dec_layer  = tf.keras.layers.Dense(
            self.hparams.embedding_dim, name = 'dec_layer'
        )
        self.enc_layer  = tf.keras.layers.Dense(
            self.hparams.embedding_dim, name = 'enc_layer'
        )
        
        self.out_layer  = tf.keras.layers.Dense(
            self.hparams.vocab_size, name = 'output_layer'
        )
        
        self.activation = get_activation(self.hparams.activation)
        self.dropout    = tf.keras.layers.Dropout(self.hparams.drop_rate) if self.hparams.drop_rate > 0. else None
    
    @property
    def vocab_size(self):
        return self.hparams.vocab_size
    
    def change_vocabulary(self, new_vocab, ** kwargs):
        self.hparams.vocab_size = len(new_vocab)
        self.out_layer.units    = self.vocab_size
        self.out_layer.build((None, None, None, self.hparams.embedding_dim))

    def call(self,
             encoder_outputs,
             decoder_outputs,
             encoder_lengths    = None,
             decoder_lengths    = None,
             training   = False,
             ** kwargs
            ):
        return self.joint(encoder_outputs, decoder_outputs, training = training)
    
    @timer
    def joint(self, encoder_outputs, decoder_outputs, training = False):
        f = self.enc_layer(encoder_outputs)
        f = tf.expand_dims(f, axis = 2)
        
        g = self.dec_layer(decoder_outputs)
        g = tf.expand_dims(g, axis = 1)
        
        out = f + g
        if self.activation is not None: out = self.activation(out)
        if self.dropout is not None: out = self.dropout(out, training = training)
        out = self.out_layer(out)
        
        return out

    def get_config(self):
        return self.hparams.get_config()
    
    @classmethod
    def from_config(cls, config, custom_objects = None):
        return cls(** config)
    
class RNNT(tf.keras.Model):
    default_hparams = HParamsRNNT
    _attr_to_set    = ['vocab_size']

    encoder_class   = None
    decoder_class   = RNNTLSTMDecoder
    joint_class     = RNNTJointNet
    
    def __init__(self, vocab_size, name = 'rnnt', ** kwargs):
        super().__init__(name = name)
        
        self.hparams = self.default_hparams.extract({
            ** kwargs,
            'vocab_size' : vocab_size,
            'decoder_vocab_size'    : vocab_size,
            'joint_vocab_size'      : vocab_size
        })
        for config in self._attr_to_set:
            setattr(self, config, self.hparams[config])
        
        self.encoder    = self.encoder_class(
            ** self.hparams.get_config(prefix = 'encoder'), name = 'encoder'
        )
        self.decoder    = self.decoder_class(
            ** self.hparams.get_config(prefix = 'decoder'), name = 'decoder'
        )
        
        self.joint_net  = self.joint_class(
            ** self.hparams.get_config(prefix = 'joint'), name = 'joint_net'
        )
    
    def _build(self):
        self(self.dummy_inputs, training = False)
    
    @property
    def sos_token(self):
        return tf.cast(self.hparams.sos_token, tf.int32)
    
    @property
    def pad_token(self):
        return tf.cast(self.hparams.pad_token, tf.int32)
    
    @property
    def dummy_inputs(self):
        batch_size, in_seq_len, out_seq_len = 2, 128, 16
        
        enc_inp = self.encoder.dummy_inputs
        dec_inp = [
            tf.ones((batch_size, out_seq_len), dtype = tf.int32),
            tf.fill((batch_size,), out_seq_len)
        ]
        
        return enc_inp, dec_inp
    
    @property
    def state_signature(self):
        return self.decoder.state_signature
    
    def change_vocabulary(self, new_vocab, sos_token = -1, pad_token = -1, ** kwargs):
        self.decoder.change_vocabulary(new_vocab, ** kwargs)
        self.joint_net.change_vocabulary(new_vocab, ** kwargs)
        
        self.vocab_size  = self.decoder.vocab_size
        self.hparams.update({
            'sos_token' : sos_token if sos_token != -1 else pad_token,
            'pad_token' : pad_token if pad_token != -1 else sos_token,
            'vocab_size'    : self.vocab_size,
            'decoder_vocab_size'    : self.vocab_size,
            'joint_vocab_size'      : self.vocab_size
        })
        
        
    
    def get_initial_state(self, inputs):
        return self.decoder.get_initial_state(inputs)

    @tf.function(reduce_retracing = True)
    def decode(self,
               inputs,
               input_length,
               encoder_output,
               encoder_length,
               initial_state    = None,
               training = False,
               return_state = False,
               ** kwargs
              ):
        tokens = inputs
        if isinstance(inputs, (list, tuple)):
            tokens, input_length = inputs
        
        decoder_output, decoder_state = self.decoder(
            tokens,
            input_length    = input_length,
            training    = training,
            initial_state   = initial_state,
            
            ** {k[8:] : v for k, v in kwargs.items() if k.startswith('decoder_')},
            ** kwargs
        )
        
        output = self.joint_net(
            encoder_output,
            decoder_output,
            encoder_lengths = encoder_length,
            decoder_lengths = input_length,
            training    = training
        )
        return output if not return_state else (output, decoder_state)

    def call(self,
             inputs,
             input_length   = None,
             speaker_embedding  = None,
             decoder_input  = None,
             decoder_input_length   = None,
             initial_state  = None,
             
             training   = False,
             
             return_state       = False,
             ** kwargs
            ):
        encoder_input = inputs
        if isinstance(inputs, (list, tuple)):
            if len(inputs) == 2:
                encoder_input, decoder_input = inputs
            else:
                encoder_input, decoder_input = inputs[:-2], inputs[-2:]
        
        encoder_output = self.encoder(
            encoder_input,
            input_length    = input_length,
            speaker_embedding   = speaker_embedding,
            training    = training,
            
            ** {k[8:] : v for k, v in kwargs.items() if k.startswith('encoder_')},
            ** kwargs
        )
        if isinstance(encoder_output, (list, tuple)):
            encoder_output, input_length = encoder_output
        
        return self.decode(
            decoder_input,
            input_length    = decoder_input_length,
            encoder_output  = encoder_output,
            encoder_length  = input_length,
            initial_state   = initial_state,
            training    = training,
            return_state    = return_state,
            ** kwargs
        )

    def infer(self,
              inputs    = None,
              input_length   = None,
              speaker_embedding = None,
              encoder_output    = None,
              
              tokens  = None,
              tokens_length   = None,
              
              blank_mask    = None,
              initial_state = None,
              training  = False,
              
              ** kwargs
             ):
        assert inputs is not None or encoder_output is not None
        
        if encoder_output is None:
            encoder_output = self.encoder(
                inputs,
                input_length    = input_length,
                speaker_embedding   = speaker_embedding,
                training    = training,

                ** {k[8:] : v for k, v in kwargs.items() if k.startswith('encoder_')},
                ** kwargs
            )
            if isinstance(encoder_output, (list, tuple)):
                encoder_output, input_length = encoder_output
        
        return transducer_infer(
            self,
            tokens  = tokens,
            input_length    = tokens_length,
            speaker_embedding   = speaker_embedding,
            initial_state   = initial_state,

            encoder_output  = encoder_output,
            encoder_output_length   = input_length,
            
            blank_mask  = blank_mask,
            training    = training,
            
            sos_token   = self.sos_token,
            blank_token = self.pad_token,
            
            ** {k[8:] : v for k, v in kwargs.items() if k.startswith('decoder_')},
            ** kwargs
        )
    
    def get_config(self):
        return self.hparams.get_config()
    
    @classmethod
    def from_config(cls, config, custom_objects = None):
        return cls(** config)


_rnnt_classes   = {
    'RNNTLSTMDecoder'   : RNNTLSTMDecoder,
    'RNNTJointNet'  : RNNTJointNet,
    'RNNT'      : RNNT
}

custom_functions    = {
    ** _rnnt_classes
}

custom_objects  = {
    ** _rnnt_classes
}