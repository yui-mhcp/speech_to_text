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
import logging
import numpy as np
import keras.ops as K

from tqdm import tqdm
from functools import cached_property

from utils import load_json, dump_json
from loggers import timer, time_logger
from utils.keras_utils import TensorSpec, ops, graph_compile
from models.stt.base_stt import BaseSTT
from utils.text.text_encoder import WHISPER_LANGUAGES
from custom_architectures.transformers_arch import whisper_arch
from utils.text import remove_tokens, remove_batch_tokens, remove_slice_tokens, process_model_output

def add_batch_index(indices, batch_size, mask = None):
    if mask is None:
        batch_indexes = ops.range(batch_size)
    else:
        indexes = ops.where(mask)
        indexes = indexes[0] if isinstance(indexes, list) else indexes[:, 0]
        batch_indexes = ops.cast(indexes, 'int32')
    
    return ops.stack([
        ops.repeat(batch_indexes, ops.shape(indices)[0]),
        ops.tile(indices, [ops.shape(batch_indexes)[0]])
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
            kwargs.setdefault('mel_fn_config',          {'n_mel_channels' : 128 if 'large' in pretrained else 80})
            kwargs.setdefault('max_input_length',       3000)
            kwargs.setdefault('use_fixed_length_input', True)
            kwargs.setdefault('pretrained_name',        pretrained)
        
        kwargs.update({
            'audio_format'  : 'mel',
            'architecture'  : 'Whisper'
        })
        super().__init__(lang = lang, pretrained = pretrained, ** kwargs)
        
        self.trim_kwargs['read_method'] = 'read_ffmpeg'
        
        self._lang_to_idx   = {
            v.strip('<|>') : i for i, v in enumerate(self.vocab)
            if v.startswith('<|') and v.strip('<|>') in WHISPER_LANGUAGES
        }
        self._idx_to_lang   = {i : v for v, i in self._lang_to_idx.items()}
        
        t = list(self.non_speech_token_indexes) + list(self.special_token_indexes)

        self.remove_tokens              = ops.cast(t, 'int32')
        self.remove_tokens_with_space   = ops.cast([
            self.text_encoder[' '], self.eos_token_idx
        ] + t, 'int32')
        
        #self._logits_filter = get_filter(self)
        self._logits_filter = None

    def build(self, pretrained = None, stt_model = None, ** kwargs):
        if stt_model is not None: return super().build(stt_model = stt_model)
    
        if pretrained is not None and 'stt_model' not in kwargs:
            super(BaseSTT, self).build(
                stt_model = whisper_arch.Whisper.from_pretrained(
                    pretrained = pretrained, decoder_eos_token = self.eos_token_idx, ** kwargs
                )
            )
        else:
            super().build(** kwargs)
    
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

    @cached_property
    def compiled_detect_language(self):
        mel_signature = self.infer_signature
        
        return graph_compile(
            self._detect_language,
            prefer_xla  = True,
            input_signature = [
                mel_signature,
                TensorSpec(
                    shape = (None, mel_signature.shape[1], self.model.hparams.encoder_embedding_dim),
                    dtype = 'float32'
                ),
                TensorSpec(shape = (None, None), dtype = 'int32')
            ]
        )
    
    def _detect_language(self, mel = None, encoder_output = None, tokens = None, training = False):
        if encoder_output is None:
            encoder_output = self.model.encoder(mel, training = training)
        
        pred = self.model.decoder(
            tokens, encoder_output = encoder_output, training = training
        )
        return K.softmax(ops.take_along_axis(
            pred[:, -1, :], ops.convert_to_tensor(self.language_indexes, 'int32')[:, None], 1
        ), axis = -1)
    
    @timer(name = 'inference', log_if_root = False)
    def infer(self,
              inputs,
              training  = False,
              *,
              
              lang  = None,
              task  = None,
              tokens    = None,
              prev_tokens   = None,
              
              decode    = False,
              
              ** kwargs
             ):
        kwargs['encoder_output_lengths']    = 1500
        kwargs.setdefault('max_length',     self.max_output_length)
        kwargs.setdefault('logits_filter',  self._logits_filter)
        
        if ops.rank(inputs) == 2: inputs = ops.expand_dims(inputs, axis = 0)

        if tokens is None:
            #if lang is None: lang = [out[0] for out in self.detect_language(inputs)]
            tokens = self.get_start_tokens(lang = lang, task = task)
        
        if ops.rank(tokens) == 1: tokens = ops.expand_dims(tokens, axis = 0)
        
        if len(tokens) == 1 and len(inputs) > 1:
            tokens = ops.tile(tokens, [len(inputs), 1])
        
        if prev_tokens is not None and len(prev_tokens) > 0 and self.start_of_prev_token_idx != -1:
            if ops.rank(prev_tokens) == 1:   prev_tokens = ops.expand_dims(prev_tokens, axis = 0)
            elif ops.rank(prev_tokens) == 3: prev_tokens = prev_tokens[:, 0, :]
            
            tokens = ops.concat([
                np.full((len(tokens), 1), self.start_of_prev_token_idx),
                prev_tokens[:, - (kwargs['max_length'] // 2 - 1) :],
                tokens
            ], axis = 1)
        
        output = self.compiled_infer(inputs, tokens = tokens, training = training, ** kwargs)

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
            mel = self.get_input(audio, pad_or_trim = True)
            if ops.rank(mel) == 2: mel = ops.expand_dims(mel, axis = 0)

            tokens  = ops.fill((mel.shape[0], 1), self.sos_token_idx)

        probs   = self.compiled_detect_language(mel = mel, tokens = tokens)
        probs   = ops.convert_to_numpy(probs)
        
        return [(
            self.languages[np.argmax(probs_i)],
            {lang : p for lang, p in zip(self.languages, probs_i)}
        ) for probs_i in probs]

    def predict_segment(self,
                        mel,
                        start,
                        end,
                        *,
                        
                        lang    = None,
                        verbose = True,
                        force_detect_language   = False,
                        condition_on_previous_text = True,
                        
                        ** kwargs
                       ):
        if end is None: end = self._get_sample_time(len(mel))
        
        kwargs['verbose'] = verbose
        
        mel = mel[self._get_sample_index(start) : self._get_sample_index(end)]
        
        n_frames    = mel.shape[0]
        segment_duration = self._get_sample_time(self.max_input_length)

        seek    = kwargs.pop('seek', 0)
        prev_seek   = seek
        input_stride    = 2 # 3000 // 1500
        time_precision  = self._get_sample_time(input_stride)

        all_tokens, all_segments = [], []
        with tqdm(total = n_frames, unit = 'frames', disable = verbose != 1) as pbar:
            while seek < n_frames:
                with time_logger.timer('segment processing'):
                    prompt = all_tokens if condition_on_previous_text else None

                    segment = self.pad_or_trim(mel[seek :])

                if lang is None and force_detect_language:
                    lang = self.detect_language(segment)[0][0]
                
                result  = self.infer(segment, prev_tokens = prompt, lang = lang, ** kwargs)
                tokens  = process_model_output(result)[0]
                if isinstance(tokens, list): tokens = tokens[0]

                if lang is None:
                    lang    = self._idx_to_lang.get(tokens[0], None)
                    tokens  = tokens[2:]
                
                with time_logger.timer('post_processing'):
                    timestamp_offset = self._get_sample_time(seek)

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
                                lang    = lang,
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
                            lang    = lang,
                            ** kwargs
                        )

                        seek += segment.shape[0]
                        all_tokens.extend(tokens)

                    # update progress bar
                    pbar.update(min(n_frames, seek) - prev_seek)
                    prev_seek = seek

        for segment in all_segments:
            segment.update({
                'start' : segment['start'] + start,
                'end'   : min(end, segment['end'] + start)
            })
            segment['time'] = segment['end'] - segment['start']
        
        return all_segments

def timestamp_filter(self, scores, tokens, to_remove, state, max_initial_timestamp = 1, ** _):
    if state.state is None:
        # suppress generating non-timestamp tokens at the beginning
        to_remove = ops.concat([
            to_remove, ops.range(self.timestamp_begin_idx)
        ], axis = -1)

        # apply the `max_initial_timestamp` option
        if max_initial_timestamp > 0:
            to_remove = ops.concat([
                to_remove, ops.range(self.timestamp_begin_idx + max_initial_timestamp, ops.shape(scores)[-1])
            ], axis = -1)

        scores = remove_batch_tokens(scores, to_remove)
    else:
        batch_size = ops.shape(scores)[0]

        last_was_timestamp          = tokens[:, -1] >= self.timestamp_begin_idx
        penultimate_was_timestamp   = ops.cond(
            state.t < 2,
            lambda: ops.ones((ops.shape(tokens)[0], ), dtype = 'bool'),
            lambda: tokens[:, -2] >= self.timestamp_begin_idx
        )

        to_remove_batch = add_batch_index(to_remove, batch_size)

        if ops.reduce_any(last_was_timestamp):
            last_but_not_penultimate    = ops.logical_and(
                last_was_timestamp, ops.logical_not(penultimate_was_timestamp)
            )
            last_and_penultimate    = ops.logical_and(
                last_was_timestamp, penultimate_was_timestamp
            )
            if ops.reduce_any(last_but_not_penultimate):
                to_remove_batch = ops.concat([
                    to_remove_batch,
                    add_batch_index(ops.range(self.eos_token_idx), batch_size, last_but_not_penultimate)
                ], axis = 0)

            if ops.reduce_any(last_and_penultimate):
                to_remove_batch = ops.concat([
                    to_remove_batch,
                    add_batch_index(ops.range(self.timestamp_begin_idx, ops.shape(scores)[-1]), batch_size, last_and_penultimate)
                ], axis = 0)

        scores = remove_tokens(scores, to_remove_batch)

        # if sum of probability over timestamps is above any other token, sample timestamp
        logits  = K.log_softmax(scores)

        timestamp_logits = K.logsumexp(logits[:, self.timestamp_begin_idx :], axis = -1)
        max_text_logits  = ops.reduce_max(logits[:, : self.timestamp_begin_idx], axis = -1)

        timestamp_over_text = timestamp_logits > max_text_logits
        if ops.reduce_any(timestamp_over_text):
            scores = remove_tokens(
                scores, add_batch_index(ops.range(self.timestamp_begin_idx), batch_size, timestamp_over_text)
            )

    return scores

def filter_logits(self, scores, tokens, state, ** _):
    to_remove = self.remove_tokens_with_space if state.state is None else self.remove_tokens
    return timestamp_filter(self, scores, tokens[:, :state.t], to_remove, state)

def get_filter(self):
    return lambda * args, ** kwargs: filter_logits(self, * args, ** kwargs)
