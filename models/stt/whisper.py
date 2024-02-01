# Copyright (C) 2022-now yui-mhcp project's author. All rights reserved.
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
import logging
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from functools import cached_property

from utils import load_json, dump_json
from loggers import timer, time_logger
from custom_layers import log_softmax
from models.stt.base_stt import BaseSTT
from utils.text.text_encoder import WHISPER_LANGUAGES
from custom_architectures.transformers_arch import whisper_arch
from utils.text import remove_tokens, remove_batch_tokens, remove_slice_tokens

def add_batch_index(indices, batch_size, mask = None):
    if mask is None:
        batch_indexes = tf.range(batch_size)
    else:
        batch_indexes = tf.cast(tf.where(mask)[:, 0], tf.int32)
    
    return tf.stack([
        tf.repeat(batch_indexes, tf.shape(indices)[0]),
        tf.tile(indices, [tf.shape(batch_indexes)[0]])
    ], axis = 1)

class Whisper(BaseSTT):
    def __init__(self, lang = 'multi', pretrained = 'base', ** kwargs):
        if pretrained:
            self.trim_kwargs = {'normalize' : 32768.}
            
            lang = 'en' if 'en' in pretrained else 'multi'
            kwargs['audio_rate'] = 16000
            kwargs.setdefault('text_encoder',           'whisper')
            kwargs.setdefault('text_encoder_config',    {'multilingual' : lang == 'multi'})
            kwargs.setdefault('mel_fn',                 'WhisperSTFT')
            kwargs.setdefault('mel_fn_config',          {})
            kwargs.setdefault('max_input_length',       3000)
            kwargs.setdefault('use_fixed_length_input', True)
            kwargs.setdefault('pretrained_name',        pretrained)
        
        kwargs.update({
            'audio_format'      : 'mel',
            'architecture_name' : 'Whisper'
        })
        super().__init__(lang = lang, pretrained = pretrained, ** kwargs)
        
        self.trim_kwargs['read_method'] = 'read_ffmpeg'
        
        self._lang_to_idx   = {
            v.strip('<|>') : i for i, v in enumerate(self.vocab)
            if v.startswith('<|') and v.strip('<|>') in WHISPER_LANGUAGES
        }
        self._idx_to_lang   = {i : v for v, i in self._lang_to_idx.items()}
        
        t = list(self.non_speech_token_indexes) + list(self.special_token_indexes)

        self.remove_tokens              = tf.cast(t, tf.int32)
        self.remove_tokens_with_space   = tf.cast([
            self.text_encoder[' '], self.eos_token_idx
        ] + t, tf.int32)
        
        self._logits_filter = get_filter(self)

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
        return '<|nospeech|>' if '<|nospeech|>' in self.text_encoder else '<|nocaptions|>'
    
    @property
    def timestamp_begin_idx(self):
        return self.vocab_size
    
    @cached_property
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
    
    @cached_property
    def language_indexes(self):
        return list(self._idx_to_lang.keys())

    @cached_property
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
            '<|startoftranscript|>', '<|startoflm|>', '<|startofprev|>', self.nospeech_token,
            '<|notimestamps|>'
        ]

    @cached_property
    def special_token_indexes(self):
        return [self.text_encoder[token] for token in self.special_tokens]

    @timer(name = 'language detection')
    @tf.function(reduce_retracing = True)
    def _detect_language(self, mel = None, encoder_output = None, tokens = None, training = False):
        if encoder_output is None: encoder_output = self.stt_model.encoder(mel, training = training)
        pred    = self.stt_model.decoder(
            tokens, encoder_output = encoder_output, training = training
        )
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
        kwargs.setdefault('max_length',     self.max_output_length)
        kwargs.setdefault('logits_filter',  self._logits_filter)
        
        if len(tf.shape(inputs)) == 2: inputs = tf.expand_dims(inputs, axis = 0)

        if tokens is None:
            if lang is None: lang = [out[0] for out in self.detect_language(inputs)]
            tokens = tf.cast(self.get_start_tokens(lang = lang, task = task), tf.int32)
        
        if len(tf.shape(tokens)) == 1: tokens = tf.expand_dims(tokens, axis = 0)
        
        if len(tokens) == 1 and len(inputs) > 1:
            tokens = tf.tile(tokens, [len(inputs), 1])
        
        if prev_tokens is not None and len(prev_tokens) > 0 and self.start_of_prev_token_idx != -1:
            prev_tokens = tf.cast(prev_tokens, tf.int32)
            if len(tf.shape(prev_tokens)) == 1: prev_tokens = tf.expand_dims(prev_tokens, axis = 0)
            if len(tf.shape(prev_tokens)) == 3: prev_tokens = prev_tokens[:, 0]
            tokens = tf.concat([
                tf.fill((len(tokens), 1), self.start_of_prev_token_idx),
                prev_tokens[:, - (kwargs['max_length'] // 2 - 1) :],
                tokens
            ], axis = -1)
        
        output = self.stt_model.infer(inputs, tokens = tokens, training = training, ** kwargs)

        return self.decode_output(output) if decode else output

    def get_start_tokens(self, lang = None, task = None):
        if isinstance(lang, (list, tuple)):
            if not isinstance(task, (list, tuple)): task = [task] * len(lang)
            return [self.get_start_tokens(l, t) for l, t in zip(lang, task)]
        
        tokens  = [self.sos_token_idx]
        if lang is None: return tokens
        
        tokens.extend([
            self._lang_to_idx[lang],
            self.translate_token_idx if task == 'translate' else self.transcribe_token_idx
        ])
        return tokens
    
    @timer
    def detect_language(self, audio):
        with time_logger.timer('pre_processing'):
            mel     = self.get_input(audio, pad_or_trim = True)
            if len(mel.shape) == 2: mel = tf.expand_dims(mel, axis = 0)

            tokens  = tf.fill((mel.shape[0], 1), self.sos_token_idx)

        probs   = self._detect_language(mel = mel, tokens = tokens)
        
        return [(
            self.languages[np.argmax(probs_i)],
            {lang : p for lang, p in zip(self.languages, probs_i)}
        ) for probs_i in probs.numpy()]

    def predict_segment(self, mel, start = 0, end = None, lang = None, verbose = True, condition_on_previous_text = True, ** kwargs):
        kwargs['verbose'] = verbose
        
        mel = mel[self._get_sample_index(start) : self._get_sample_index(end)]
        
        n_frames    = mel.shape[0]
        segment_duration = float(
            self.max_input_length * self.mel_fn.hop_length / self.audio_rate
        )

        seek    = 0
        prev_seek   = 0
        input_stride    = self.max_input_length // self.stt_model.encoder.max_input_length
        time_precision  = float(input_stride * self.mel_fn.hop_length / self.audio_rate)

        all_tokens, all_segments = [], []
        with tqdm(total = n_frames, unit = 'frames', disable = verbose != 1) as pbar:
            while seek < n_frames:
                with time_logger.timer('segment processing'):
                    prompt = all_tokens if condition_on_previous_text else None

                    segment = self.pad_or_trim(mel[seek :])

                if lang is None: lang = self.detect_language(segment)[0][0]
                result  = self.infer(segment, prev_tokens = prompt, lang = lang, ** kwargs)

                with time_logger.timer('post_processing'):
                    timestamp_offset = float(seek * self.mel_fn.hop_length / self.audio_rate)

                    tokens  = result.tokens[0].numpy()
                    timestamp_tokens    = tokens >= self.timestamp_begin_idx
                    consecutive         = np.where(np.logical_and(
                        timestamp_tokens[:-1], timestamp_tokens[1:]
                    ))[0] + 1
                    # if the output contains two consecutive timestamp tokens
                    if len(consecutive) > 0:
                        last_slice = 0
                        for current_slice in consecutive:
                            sliced_tokens = tokens[last_slice : current_slice]
                            start_timestamp_position = (
                                sliced_tokens[0] - self.timestamp_begin_idx
                            )
                            end_timestamp_position = (
                                sliced_tokens[-1] - self.timestamp_begin_idx
                            )
                            sliced_tokens = sliced_tokens[1 : -1]
                            self._add_segment(
                                all_segments,
                                segment = segment,
                                start   = timestamp_offset + start_timestamp_position * time_precision,
                                end     = timestamp_offset + end_timestamp_position * time_precision,
                                tokens  = sliced_tokens[sliced_tokens < self.eos_token_idx],
                                result  = result,
                                ** kwargs
                            )
                            last_slice = current_slice

                        last_timestamp_position = (
                            tokens[last_slice - 1] - self.timestamp_begin_idx
                        )
                        seek += last_timestamp_position * input_stride
                        all_tokens.extend(tokens[: last_slice + 1])
                    else:
                        duration    = segment_duration
                        timestamps  = tokens[timestamp_tokens]
                        if len(timestamps) > 0 and timestamps[-1] != self.timestamp_begin_idx:
                            # no consecutive timestamps but it has a timestamp; use the last one.
                            # single timestamp at the end means no speech after the last timestamp.
                            last_timestamp_position = timestamps[-1] - self.timestamp_begin_idx
                            duration = float(last_timestamp_position) * time_precision

                        self._add_segment(
                            all_segments,
                            segment = segment,
                            start   = timestamp_offset,
                            end     = timestamp_offset + duration,
                            tokens  = tokens[tokens < self.eos_token_idx],
                            result  = result,
                            ** kwargs
                        )

                        seek += segment.shape[0]
                        all_tokens.extend(tokens)

                    # update progress bar
                    pbar.update(min(n_frames, seek) - prev_seek)
                    prev_seek = seek
                    
        return all_segments

def timestamp_filter(self, scores, tokens, to_remove, state, max_initial_timestamp = 1, ** _):
    if state.state is None:
        # suppress generating non-timestamp tokens at the beginning
        to_remove = tf.concat([
            to_remove, tf.range(self.timestamp_begin_idx)
        ], axis = -1)

        # apply the `max_initial_timestamp` option
        if max_initial_timestamp > 0:
            to_remove = tf.concat([
                to_remove, tf.range(self.timestamp_begin_idx + max_initial_timestamp, tf.shape(scores)[-1])
            ], axis = -1)

        scores = remove_batch_tokens(scores, to_remove)
    else:
        batch_size = tf.shape(scores)[0]

        last_was_timestamp          = tokens[:, -1] >= self.timestamp_begin_idx
        penultimate_was_timestamp   = tf.cond(
            state.step < 2,
            lambda: tf.ones((tf.shape(tokens)[0], ), dtype = tf.bool),
            lambda: tokens[:, -2] >= self.timestamp_begin_idx
        )

        to_remove_batch = add_batch_index(to_remove, batch_size)

        if tf.reduce_any(last_was_timestamp):
            last_but_not_penultimate    = tf.logical_and(
                last_was_timestamp, tf.logical_not(penultimate_was_timestamp)
            )
            last_and_penultimate    = tf.logical_and(
                last_was_timestamp, penultimate_was_timestamp
            )
            if tf.reduce_any(last_but_not_penultimate):
                to_remove_batch = tf.concat([
                    to_remove_batch,
                    add_batch_index(tf.range(self.eos_token_idx), batch_size, last_but_not_penultimate)
                ], axis = 0)

            if tf.reduce_any(last_and_penultimate):
                to_remove_batch = tf.concat([
                    to_remove_batch,
                    add_batch_index(tf.range(self.timestamp_begin_idx, tf.shape(scores)[-1]), batch_size, last_and_penultimate)
                ], axis = 0)

        scores = remove_tokens(scores, to_remove_batch)

        # if sum of probability over timestamps is above any other token, sample timestamp
        logits  = log_softmax(scores)

        timestamp_logits = tf.math.reduce_logsumexp(logits[:, self.timestamp_begin_idx :], axis = -1)
        max_text_logits  = tf.reduce_max(logits[:, : self.timestamp_begin_idx], axis = -1)

        timestamp_over_text = timestamp_logits > max_text_logits
        if tf.reduce_any(timestamp_over_text):
            scores = remove_tokens(
                scores, add_batch_index(tf.range(self.timestamp_begin_idx), batch_size, timestamp_over_text)
            )

    return scores

def filter_logits(self, scores, tokens, state, ** _):
    to_remove = self.remove_tokens_with_space if state.state is None else self.remove_tokens
    return timestamp_filter(self, scores, tokens[:, :state.t], to_remove, state)

def get_filter(self):
    return lambda * args, ** kwargs: filter_logits(self, * args, ** kwargs)
