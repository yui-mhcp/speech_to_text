
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

import logging
import collections
import tensorflow as tf

from loggers import timer
from utils import get_object
from custom_layers import log_softmax

time_logger = logging.getLogger('timer')

TransducerInferenceOutput = collections.namedtuple(
    "TransducerInferenceOutput", [
        "tokens",
        "lengths",
        "scores",
        "blank_mask",
        "state"
    ]
)

TransducerInferenceState   = collections.namedtuple(
    "TransducerInferenceState", [
        "tokens",
        "lengths",
        "state",
        "blank_mask",
        "scores",
        
        "frame_index",
        "successive_skip",
        "n_added"
    ]
)

def get_shape_invariant(model, encoder_output = None, ** kwargs):
    return TransducerInferenceState(
        tokens          = tf.TensorSpec(shape = (None, None),       dtype = tf.int32),
        lengths         = tf.TensorSpec(shape = (None,),            dtype = tf.int32),
        state           = model.state_signature,
        blank_mask      = tf.TensorSpec(shape = (None, None),       dtype = tf.float32),
        scores          = tf.TensorSpec(shape = (None,),            dtype = tf.float32),
        
        frame_index     = tf.TensorSpec(shape = (None,),            dtype = tf.int32),
        successive_skip = tf.TensorSpec(shape = (None,),            dtype = tf.int32),
        n_added         = tf.TensorSpec(shape = (None,),            dtype = tf.int32)
    )

@timer
def transducer_infer(model, * args, method = 'greedy', ** kwargs):
    return get_object(_inference_methods, method, * args, model = model, ** kwargs)

def _transducer_infer(model,
                      tokens    = None,
                      input_length  = None,
                      blank_mask    = None,
                      encoder_output    = None,
                      encoder_output_length = None,
                      initial_state     = None,

                      training  = False,
                      use_cache = False,

                      sos_token     = None,
                      blank_token   = None,
                      max_token_per_frame   = 10,
                      restart_after_succ    = -1,
                      use_sampling = False,

                      temperature   = 1.,
                      length_temperature    = 0.,
                      length_power  = 1.,
                      
                      return_state  = False,
                      
                      ** kwargs
                     ):
    @timer
    def cond(tokens, lengths, state, blank_mask, scores, frame_index, successive_skip, n_added):
        return not tf.reduce_all(_get_finished_mask(
            frame_index = frame_index, encoder_output_length = encoder_output_length
        ))
        #return tf.reduce_any(tf.math.logical_or(
        #    frame_index < encoder_output_length - 1,
        #    blank_mask[:, -1] == 0.
        #))
    
    @timer
    def body(tokens, lengths, state, blank_mask, scores, frame_index, successive_skip, n_added):
        input_tokens    = tokens[:, -1:]
        frames          = _get_frame(encoder_output, encoder_output_length, frame_index)
        
        output, next_state = model.decode(
            input_tokens,
            input_length    = lengths,
            encoder_output  = frames,
            encoder_length  = tf.cast(encoder_output_length - frame_index > 0, tf.int32),
            initial_state   = state,
            training    = training,
            
            return_state    = True,
            
            ** kwargs
        )
        
        logits      = _compute_logits(
            output, temperature = temperature, length_temperature = length_temperature
        )
        
        next_token  = _select_next_token(
            logits, n = 1, previous = tokens, use_sampling = use_sampling
        )
        scores      = scores + tf.gather(logits, next_token, batch_dims = 1)
        
        time_logger.start_timer('update variables')
        
        is_blank    = tf.math.equal(next_token, blank_token)
        next_token  = tf.expand_dims(next_token, axis = 1)
        
        tok_mask    = tf.cast(tf.math.logical_not(is_blank), tf.int32)
        n_added     = (n_added + 1) * tok_mask
        lengths     = lengths + tok_mask
        successive_skip = (successive_skip + 1) * tf.cast(is_blank, tf.int32)

        next_frame  = tf.math.logical_or(
            is_blank, n_added >= max_token_per_frame
        )
        frame_index = frame_index + tf.cast(next_frame, tf.int32)
        
        is_blank    = tf.expand_dims(is_blank, axis = 1)

        next_state  = keep_not_blank(state, next_state, is_blank)
        next_token  = keep_not_blank(input_tokens, next_token, is_blank)
        
        time_logger.stop_timer('update variables')

        if restart_after_succ > 1:
            should_restart  = tf.expand_dims(successive_skip == restart_after_succ, axis = 1)
            if tf.reduce_any(should_restart):
                next_token  = keep_not_blank(next_token, _zero_token, should_restart)
                next_state  = keep_not_blank(next_state, _zero_state, should_restart)

        tokens      = tf.concat([tokens, next_token], axis = -1)
        blank_mask  = tf.concat([
            blank_mask, tf.cast(is_blank, blank_mask.dtype)
        ], axis = -1)


        return TransducerInferenceState(
            tokens  = tokens,
            lengths = lengths,
            state   = next_state,
            blank_mask  = blank_mask,
            scores  = scores,
            
            frame_index = frame_index,
            successive_skip = successive_skip,
            n_added = n_added
        )
    
    batch_size = tf.shape(encoder_output)[0]
    
    sos_created = False
    _zero_state = model.get_initial_state(encoder_output)
    _zero_token = tf.fill((batch_size, 1), sos_token)
    
    if encoder_output_length is None:
        encoder_output_length   = tf.fill((batch_size,), tf.shape(encoder_output)[1] - 1)
    else:
        encoder_output_length   = tf.minimum(encoder_output_length, tf.shape(encoder_output)[1])

    if tokens is None:
        sos_created     = True
        tokens          = tf.fill((batch_size, 1), sos_token)
        input_length    = tf.fill((batch_size,), 1)
    elif isinstance(tokens, (list, tuple)):
        tokens, input_length    = tokens
    
    if input_length is None:
        input_length    = tf.fill((batch_size,), tf.shape(tokens)[1])

    if initial_state is None:
        initial_state = _zero_state
    
    if blank_mask is None:
        blank_mask = tf.zeros((batch_size, tf.shape(tokens)[1]))
    
    outputs = tf.while_loop(
        cond    = cond,
        body    = body,
        loop_vars   = TransducerInferenceState(
            tokens  = tokens,
            lengths = input_length,
            state   = initial_state,
            blank_mask  = blank_mask,
            scores      = tf.zeros((batch_size,)),
            
            frame_index     = tf.zeros((batch_size,), dtype = tf.int32),
            successive_skip = tf.zeros((batch_size,), dtype = tf.int32),
            n_added         = tf.zeros((batch_size,), dtype = tf.int32)
        ),
        shape_invariants    = get_shape_invariant(model, encoder_output = encoder_output),
        maximum_iterations  = -1 if max_token_per_frame == -1 else max_token_per_frame * tf.shape(encoder_output)[1]
    )
    tokens, lengths, blank_mask = outputs.tokens, outputs.lengths, outputs.blank_mask
    if sos_created:
        tokens, lengths, blank_mask = tokens[..., 1:], lengths - 1, blank_mask[..., 1:]
    
    return TransducerInferenceOutput(
        tokens  = tokens,
        lengths = lengths,
        scores  = outputs.scores / (tf.cast(lengths, tf.float32) ** length_power),
        blank_mask  = blank_mask,
        state   = outputs.state if return_state else None
    )

@timer
def _get_finished_mask(frame_index, encoder_output_length):
    return frame_index >= encoder_output_length

@timer
def _get_frame(encoder_output, length, frame_index):
    return tf.expand_dims(
        tf.gather(encoder_output, tf.minimum(frame_index, length), batch_dims = 1), axis = 1
    )

@timer
def _compute_logits(output, temperature = 1., length_temperature = 0.):
    logits  = output[:, 0, 0, :]
    if temperature != 1.:
        logits = logits / temperature
    
    if length_temperature != 0.:
        _lengths = tf.cast(lengths + 1, tf.float32)
        if length_temperature == -1.:
            temp = tf.maximum(tf.math.log(_lengths), 1.)
        else:
            temp = _lengths ** length_temperature
        temp = 1. / temp

        logits = logits / temp
    
    return log_softmax(logits, axis = -1)

@timer
def _select_next_token(scores, n = 1, previous = None, use_sampling = False, dtype = tf.int32):
    if not use_sampling:
        if n == 1: return tf.argmax(scores, axis = -1, output_type = dtype)
        return tf.cast(tf.nn.top_k(scores, k = n).indices, dtype)
    
    raise NotImplementedError()

@timer
def keep_not_blank(val, next_val, blank_indexes):
    def _keep_not_blank(v1, v2):
        return tf.where(blank_indexes, v1, v2)
    
    _one = tf.cast(1, blank_indexes.dtype)
    if isinstance(val, tf.Tensor):
        return _keep_not_blank(val, next_val)
    return tf.nest.map_structure(
        _keep_not_blank, val, next_val
    )

    
_inference_methods  = {
    'greedy'    : _transducer_infer,
    'sample'    : lambda * args, ** kwargs: _transducer_infer(* args, use_sampling = True, ** kwargs)
}