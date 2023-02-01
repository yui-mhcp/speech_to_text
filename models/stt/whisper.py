
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
import itertools
import numpy as np
import tensorflow as tf

from functools import lru_cache

from loggers import timer
from custom_layers import log_softmax
from models.stt.base_stt import BaseSTT
from utils.text.text_encoder import WHISPER_LANGUAGES
from custom_architectures.transformers_arch import whisper_arch
from utils.text import remove_tokens, remove_batch_tokens, remove_slice_tokens

time_logger = logging.getLogger('timer')

class Whisper(BaseSTT):
    def __init__(self, lang = 'multi', pretrained = 'base', ** kwargs):
        if pretrained:
            self.trim_kwargs = {'normalize' : 32768.}
            
            lang = 'en' if 'en' in pretrained else 'multi'
            kwargs['audio_rate'] = 16000
            kwargs.setdefault('text_encoder', 'whisper')
            kwargs.setdefault('text_encoder_config', {'multilingual' : lang == 'multi'})
            kwargs.setdefault('mel_fn', 'WhisperSTFT')
            kwargs.setdefault('mel_fn_config', {})
            kwargs.setdefault('pretrained_name', pretrained)
        
        kwargs.update({
            'audio_format'      : 'mel',
            'architecture_name' : 'Whisper'
        })
        super().__init__(lang = lang, pretrained = pretrained, ** kwargs)
        
        self._lang_to_idx   = {
            v.strip('<|>') : i for i, v in enumerate(self.vocab)
            if v.startswith('<|') and v.strip('<|>') in WHISPER_LANGUAGES
        }
        self._idx_to_lang   = {i : v for v, i in self._lang_to_idx.items()}

    def _build_model(self, pretrained = None, ** kwargs):
        if pretrained is not None:
            super(BaseSTT, self)._build_model(
                stt_model = whisper_arch.Whisper.from_pretrained(
                    pretrained = pretrained, decoder_eos_token = self.eos_token_idx, ** kwargs
                )
            )
        else:
            kwargs['architecture_name'] = 'Whisper'
            super()._build_model(** kwargs)

    @property
    def sos_token(self):
        return '<|startoftranscript|>'
    
    @property
    def transcribe_token(self):
        return '<|transcribe|>'
        
    @property
    def translate_token(self):
        return '<|translate|>'
    
    @property
    def start_of_prev_token(self):
        return '<|startofprev|>'
    
    @property
    def nospeech_token(self):
        return '<|nospeech|>'
    
    @property
    def timestamp_begin_idx(self):
        return self.vocab_size
    
    @property
    @lru_cache()
    def languages(self):
        return list(self._lang_to_idx.keys())

    @property
    def sos_token_idx(self):
        return self.text_encoder[self.sos_token]
    
    @property
    def transcribe_token_idx(self):
        return self.text_encoder[self.transcribe_token]
        
    @property
    def translate_token_idx(self):
        return self.text_encoder[self.translate_token]
    
    @property
    def start_of_prev_token_idx(self):
        return self.text_encoder[self.start_of_prev_token]
    
    @property
    def nospeech_token_idx(self):
        return self.text_encoder[self.nospeech_token]
    
    @property
    @lru_cache()
    def language_indexes(self):
        return list(self._idx_to_lang.keys())

    @property
    @lru_cache()
    def non_speech_token_indexes(self):
        """ Defines non-speech tokens as defined in the original openai/whisper project """
        symbols = list("\"#()*+/:;<=>@[\\]^_`{|}~「」『』")
        symbols += "<< >> <<< >>> -- --- -( -[ (' (\" (( )) ((( ))) [[ ]] {{ }} ♪♪ ♪♪♪".split()

        # symbols that may be a single token or multiple tokens depending on the tokenizer.
        # In case they're multiple tokens, suppress the first token, which is safe because:
        # These are between U+2640 and U+267F miscellaneous symbols that are okay to suppress
        # in generations, and in the 3-byte UTF-8 representation they share the first two bytes.
        miscellaneous = set("♩♪♫♬♭♮♯")
        assert all(0x2640 <= ord(c) <= 0x267F for c in miscellaneous)
        # allow hyphens "-" and single quotes "'" between words, but not at the beginning of a word
        result = {self.text_encoder[" -"], self.text_encoder[" '"]}
        for symbol in symbols + list(miscellaneous):
            for tokens in [self.text_encoder[symbol], self.text_encoder[" " + symbol]]:
                if isinstance(tokens, int):
                    result.add(tokens)
                elif symbol in miscellaneous and tokens:
                    result.add(tokens[0])

        return tuple(sorted(result))

    @property
    def special_tokens(self):
        return [
            '<|startoftranscript|>', '<|startoflm|>', '<|startofprev|>', '<|nospeech|>',
            '<|notimestamps|>'
        ]

    @property
    def special_token_indexes(self):
        return [self.text_encoder[token] for token in self.special_tokens]

    @timer
    @tf.function(reduce_retracing = True)
    def _detect_language(self, mel = None, encoder_output = None, tokens = None, training = False):
        if encoder_output is None: encoder_output = self.stt_model.encoder(mel, training = training)
        pred    = self.stt_model.decoder(
            tokens, encoder_output = encoder_output, training = training
        )[:, 0]

        return tf.nn.softmax(tf.gather(
            pred, tf.cast(self.language_indexes, tf.int32), axis = -1
        ), axis = -1)
    
    @timer(name = 'inference', log_if_root = False)
    def infer(self,
              inputs,
              tokens    = None,
              training  = False,
              lang      = None,
              task      = None,
              decode    = False,
              prev_tokens   = None,
              ** kwargs
             ):
        kwargs.setdefault('max_length', self.max_output_length)
        kwargs.setdefault('logits_filter', self.filter_logits)
        
        if len(tf.shape(inputs)) == 2: inputs = tf.expand_dims(inputs, axis = 0)

        if tokens is None:
            if lang is None: lang = self.detect_language(inputs[0])[0][0]
            tokens = self.get_start_tokens(lang = lang, task = task)
        
        if len(tf.shape(tokens)) == 1: tokens = tf.expand_dims(tokens, axis = 0)
        
        if tf.shape(tokens)[0] == 1 and tf.shape(inputs)[0] > 1:
            tokens = tf.tile(tokens, [tf.shape(inputs)[0], 1])
        
        if prev_tokens is not None and self.start_of_prev_token_idx != -1:
            if len(tf.shape(prev_tokens)) == 3: prev_tokens = prev_tokens[:, 0]
            tokens = tf.concat([
                tf.fill((tf.shape(tokens)[0], 1), self.start_of_prev_token_idx), prev_tokens, tokens
            ], axis = -1)
        
        output = self.stt_model.infer(inputs, tokens = tokens, training = training, ** kwargs)
        
        return self.decode_output(output) if decode else output

    def get_start_tokens(self, lang = None, task = None):
        tokens  = [self.sos_token_idx]
        if lang is None: return tokens
        
        tokens.extend([
            self._lang_to_idx[lang],
            self.translate_token_idx if task == 'translate' else self.transcribe_token_idx
        ])
        return tokens
    
    @tf.function(input_signature = [
        tf.TensorSpec(shape = (None, None), dtype = tf.float32),
        tf.TensorSpec(shape = (None, None), dtype = tf.int32),
        tf.TensorSpec(shape = (),           dtype = tf.int32),
        tf.TensorSpec(shape = (),           dtype = tf.int32)
    ])
    def timestamp_filter(self, scores, tokens, t = -1, max_initial_timestamp_index = 1):
        if t == 0:
            # suppress generating non-timestamp tokens at the beginning
            scores = remove_slice_tokens(scores, self.timestamp_begin_idx, remove_after = False)

            # apply the `max_initial_timestamp` option
            if max_initial_timestamp_index > 0:
                scores = remove_slice_tokens(
                    scores, self.timestamp_begin_idx + max_initial_timestamp_index, remove_after = True
                )
        else:
            last_was_timestamp          = tokens[:, -1] >= self.timestamp_begin_idx
            penultimate_was_timestamp   = t < 2 or tokens[:, -2] >= self.timestamp_begin_idx

            if tf.reduce_any(last_was_timestamp):
                last_but_not_penultimate    = tf.logical_and(
                    last_was_timestamp, tf.logical_not(penultimate_was_timestamp)
                )
                last_and_penultimate    = tf.logical_and(
                    last_was_timestamp, penultimate_was_timestamp
                )
                if tf.reduce_any(last_but_not_penultimate):
                    filtered = remove_slice_tokens(scores, self.eos_token_idx, remove_after = False)
                    scores   = tf.where(tf.expand_dims(last_but_not_penultimate, 1), filtered, scores)
                
                if tf.reduce_any(last_and_penultimate):
                    filtered = remove_slice_tokens(scores, self.timestamp_begin_idx, remove_after = True)
                    scores   = tf.where(tf.expand_dims(last_and_penultimate, 1), filtered, scores)

            # if sum of probability over timestamps is above any other token, sample timestamp
            logits  = log_softmax(scores)

            timestamp_logits = tf.math.reduce_logsumexp(logits[:, self.timestamp_begin_idx :], axis = -1)
            max_text_logits  = tf.reduce_max(logits[:, : self.timestamp_begin_idx])

            timestamp_over_text = timestamp_logits > max_text_logits
            if tf.reduce_any(timestamp_over_text):
                filtered = remove_slice_tokens(scores, self.timestamp_begin_idx, remove_after = False)
                scores   = tf.where(tf.expand_dims(timestamp_over_text, 1), filtered, scores)
        
        return scores

    @tf.function(input_signature = [
        tf.TensorSpec(shape = (None, None), dtype = tf.float32),
        tf.TensorSpec(shape = (None, None), dtype = tf.int32),
        tf.TensorSpec(shape = (),           dtype = tf.int32)
    ])
    def filter_logits(self, scores, tokens, t = -1):
        def _get_tokens(t):
            to_remove = []

            if t == 0: to_remove.extend([self.text_encoder[' '], self.eos_token_idx])
            to_remove.extend(self.non_speech_token_indexes)
            to_remove.extend(self.special_token_indexes)
            return tf.cast(to_remove, tf.int32)
        
        to_remove = tf.numpy_function(
            _get_tokens, [t], Tout = tf.int32
        )
        to_remove.set_shape([None])
        
        filetered = remove_batch_tokens(scores, to_remove)
        filetered = self.timestamp_filter(filetered, tokens, t = t)
        filetered.set_shape(scores.shape)
        
        return filetered

    def detect_language(self, audio):
        mel     = self.get_input(audio)
        mel     = pad_batch(mel, pad_value = self.pad_mel_value) if isinstance(mel, list) else tf.expand_dims(mel, axis = 0)
        
        tokens  = tf.tile(tf.expand_dims(self.get_start_tokens(), axis = 0), [tf.shape(mel)[0], 1])
        
        probs   = self._detect_language(mel = mel, tokens = tokens)
        
        return [(
            self.languages[np.argmax(probs_i)],
            {lang : p for lang, p in zip(self.languages, probs_i)}
        ) for probs_i in probs.numpy()]

    def predict(self, * args, ** kwargs):
        kwargs.setdefault('use_prev', True)
        return super().predict(* args, ** kwargs)
    