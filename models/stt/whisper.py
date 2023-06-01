
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

import os
import glob
import logging
import itertools
import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm import tqdm
from functools import lru_cache

from loggers import timer
from custom_layers import log_softmax
from models.stt.base_stt import BaseSTT
from utils.text.text_encoder import WHISPER_LANGUAGES
from custom_architectures.transformers_arch import whisper_arch
from utils.text import remove_tokens, remove_batch_tokens, remove_slice_tokens
from utils.audio.audio_io import read_ffmpeg
from utils import load_json, dump_json

time_logger = logging.getLogger('timer')

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
        
        self.trim_kwargs['read_method'] = read_ffmpeg
        
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
        kwargs.setdefault('max_length', self.max_output_length)
        kwargs.setdefault('logits_filter', self.filter_logits)
        
        if len(tf.shape(inputs)) == 2: inputs = tf.expand_dims(inputs, axis = 0)

        if tokens is None:
            if lang is None: lang = [out[0] for out in self.detect_language(inputs)]
            tokens = tf.cast(self.get_start_tokens(lang = lang, task = task), tf.int32)
        
        if len(tf.shape(tokens)) == 1: tokens = tf.expand_dims(tokens, axis = 0)
        
        if tf.shape(tokens)[0] == 1 and tf.shape(inputs)[0] > 1:
            tokens = tf.tile(tokens, [tf.shape(inputs)[0], 1])
        
        if prev_tokens is not None and len(prev_tokens) > 0 and self.start_of_prev_token_idx != -1:
            prev_tokens = tf.cast(prev_tokens, tf.int32)
            if len(tf.shape(prev_tokens)) == 1: prev_tokens = tf.expand_dims(prev_tokens, axis = 0)
            if len(tf.shape(prev_tokens)) == 3: prev_tokens = prev_tokens[:, 0]
            tokens = tf.concat([
                tf.fill((tf.shape(tokens)[0], 1), self.start_of_prev_token_idx),
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
    
    @tf.function(input_signature = [
        tf.TensorSpec(shape = (None, None), dtype = tf.float32),
        tf.TensorSpec(shape = (None, None), dtype = tf.int32),
        tf.TensorSpec(shape = (None, ),     dtype = tf.int32),
        tf.TensorSpec(shape = (),           dtype = tf.int32),
        tf.TensorSpec(shape = (),           dtype = tf.int32)
    ])
    def timestamp_filter(self, scores, tokens, to_remove, t = -1, max_initial_timestamp_index = 1):
        if t == 0:
            # suppress generating non-timestamp tokens at the beginning
            to_remove = tf.concat([
                to_remove, tf.range(self.timestamp_begin_idx)
            ], axis = -1)

            # apply the `max_initial_timestamp` option
            if max_initial_timestamp_index > 0:
                to_remove = tf.concat([
                    to_remove, tf.range(self.timestamp_begin_idx + max_initial_timestamp_index, tf.shape(scores)[-1])
                ], axis = -1)

            scores = remove_batch_tokens(scores, to_remove)
        else:
            batch_size = tf.shape(scores)[0]

            last_was_timestamp          = tokens[:, -1] >= self.timestamp_begin_idx
            penultimate_was_timestamp   = t < 2 or tokens[:, -2] >= self.timestamp_begin_idx

            to_remove = add_batch_index(to_remove, batch_size)

            if tf.reduce_any(last_was_timestamp):
                last_but_not_penultimate    = tf.logical_and(
                    last_was_timestamp, tf.logical_not(penultimate_was_timestamp)
                )
                last_and_penultimate    = tf.logical_and(
                    last_was_timestamp, penultimate_was_timestamp
                )
                if tf.reduce_any(last_but_not_penultimate):
                    to_remove = tf.concat([
                        to_remove,
                        add_batch_index(tf.range(self.eos_token_idx), batch_size, last_but_not_penultimate)
                    ], axis = 0)

                if tf.reduce_any(last_and_penultimate):
                    to_remove = tf.concat([
                        to_remove,
                        add_batch_index(tf.range(self.timestamp_begin_idx, tf.shape(scores)[-1]), batch_size, last_and_penultimate)
                    ], axis = 0)

            scores = remove_tokens(scores, to_remove)

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

    @tf.function(input_signature = [
        tf.TensorSpec(shape = (None, None), dtype = tf.float32),
        tf.TensorSpec(shape = (None, None), dtype = tf.int32),
        tf.TensorSpec(shape = (),           dtype = tf.int32)
    ])
    def filter_logits(self, scores, tokens, t = -1):
        to_remove   = tf.cond(
            t == 0,
            lambda: self.remove_tokens_with_space,
            lambda: self.remove_tokens
        )
        filtered = self.timestamp_filter(scores, tokens, to_remove, t = t)
        filtered.set_shape(scores.shape)

        return filtered
    
    @timer
    def detect_language(self, audio):
        time_logger.start_timer('pre_processing')
        mel     = self.get_input(audio, pad_or_trim = True)
        if isinstance(mel, list):
            mel = tf.cast(pad_batch(mel, pad_value = self.pad_mel_value)) if len(mel) > 1 else mel[0]
        if len(mel.shape) == 2:
            mel = tf.expand_dims(mel, axis = 0)
        
        tokens  = tf.fill((mel.shape[0], 1), self.sos_token_idx)
        
        time_logger.stop_timer('pre_processing')

        probs   = self._detect_language(mel = mel, tokens = tokens)
        
        return [(
            self.languages[np.argmax(probs_i)],
            {lang : p for lang, p in zip(self.languages, probs_i)}
        ) for probs_i in probs.numpy()]

    @timer
    def predict(self,
                filenames,
                lang = None,
                
                save    = True,
                directory   = None,
                overwrite   = False,
                timestamp   = -1,
                raw_audio_dir   = None,
                raw_audio_filename  = 'audio_{}.mp3',

                condition_on_previous_text  = True,
                
                post_processing = None,
                
                verbose = False,
                ** kwargs
               ):
        ####################
        # Helper functions #
        ####################
        
        def post_process(segment, infos):
            if verbose == 2:
                print('Add segment from {} to {} with text {}'.format(
                    infos['start'], infos['end'], infos['text']
                ))
            
            if post_processing is not None:
                post_processing(segment, infos)
            
            return infos
        
        @timer
        def add_segment(segment, start, end, tokens, result):
            tokens  = [token for token in tokens if token < self.eos_token_idx]
            text    = self.decode_text(tokens) if tokens else ''

            infos ={
                "id"    : -1,
                "num"   : len(all_segments),
                "seek"  : seek,
                "start" : start,
                "end"   : end,
                "time"  : end - start,
                "text"  : text.strip(),
                "tokens"    : tokens,
                "score"     : result.score[0].numpy()
            }
            if text: all_segments.append(infos)
            
            return post_process(segment, infos)

        def should_predict(audio):
            if isinstance(audio, (dict, pd.Series)) and 'filename' in audio:
                audio = audio['filename']
            if isinstance(audio, str) and audio in predicted:
                if not overwrite or (timestamp != -1 and timestamp <= predicted[audio].get('timestamp', -1)):
                    return False
            return True
        
        def get_filename(audio):
            if isinstance(audio, (dict, pd.Series)):
                audio = audio.get('filename', None)
            if isinstance(audio, (np.ndarray, tf.Tensor)):
                return None
            elif isinstance(audio, str):
                return audio.replace(os.path.sep, '/')
            raise ValueError('Unknown audio type ({}) : {}'.format(type(audio), audio))

        ####################
        #  Initialization  #
        ####################
        
        time_logger.start_timer('initialization')
        
        if not isinstance(filenames, (list, tuple, pd.DataFrame)): filenames = [filenames]
        
        if directory is None: directory = self.pred_dir
        raw_audio_dir   = os.path.join(directory, 'audios')
        map_file    = os.path.join(directory, 'map.json')
        
        predicted = {}
        if save:
            os.makedirs(directory, exist_ok = True)
            
            predicted = load_json(map_file, default = {})
        
        
        results = [None] * len(filenames)
        duplicatas  = {}
        requested   = [(get_filename(audio), audio) for audio in filenames]
        
        inputs = []
        for i, (file, audio) in enumerate(requested):
            if not should_predict(file):
                if verbose: print('Audio {} already processed'.format(file))
                results[i] = (file, predicted[file])
                continue
            
            if isinstance(file, str):
                duplicatas.setdefault(file, []).append(i)
                
                if len(duplicatas[file]) > 1:
                    continue
            
            inputs.append((i, file, audio))

        time_logger.stop_timer('initialization')
        
        for idx, file, data in inputs:
            time_logger.start_timer('audio loading')
            
            if isinstance(file, str) and verbose:
                print('Processing file {}...'.format(file))
            
            mel = self.get_input(data, pad_or_trim = False)

            n_frames    = mel.shape[0]
            segment_duration = float(
                self.max_input_length * self.mel_fn.hop_length / self.audio_rate
            )
            
            seek    = 0
            input_stride    = self.max_input_length // self.stt_model.encoder.max_input_length
            time_precision  = float(input_stride * self.mel_fn.hop_length / self.audio_rate)

            time_logger.stop_timer('audio loading')

            seek, all_tokens, all_segments, prev_seek = 0, [], [], 0
            with tqdm(total = n_frames, unit = 'frames', disable = verbose != 1) as pbar:
                while seek < n_frames:
                    time_logger.start_timer('pre_processing')
                    
                    prompt = all_tokens if condition_on_previous_text else None

                    segment = self.pad_or_trim(mel[seek :])

                    time_logger.stop_timer('pre_processing')

                    if lang is None: lang = self.detect_language(segment)[0][0]
                    result  = self.infer(segment, prev_tokens = prompt, lang = lang, ** kwargs)
                    
                    
                    time_logger.start_timer('post_processing')

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
                            add_segment(
                                segment = segment,
                                start   = timestamp_offset + start_timestamp_position * time_precision,
                                end     = timestamp_offset + end_timestamp_position * time_precision,
                                tokens  = sliced_tokens[1 : -1],
                                result  = result
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

                        add_segment(
                            segment = segment,
                            start   = timestamp_offset,
                            end     = timestamp_offset + duration,
                            tokens  = tokens,
                            result  = result
                        )

                        seek += segment.shape[0]
                        all_tokens.extend(tokens)

                    # update progress bar
                    pbar.update(min(n_frames, seek) - prev_seek)
                    prev_seek = seek
                    
                    time_logger.stop_timer('post_processing')

            infos = {
                'filename'  : file,
                'text'  : ' '.join([seg['text'] for seg in all_segments]),
                'lang'  : lang,
                'alignment' : all_segments
            }
            
            if file and file in duplicatas:
                for idx in duplicatas[file]:
                    results[idx] = (data, infos)
            else:
                results[i] = (data, infos)
            
            if save:
                if not file:
                    time_logger.start_timer('saving audios')
                    
                    os.makedirs(raw_audio_dir, exist_ok = True)
                    file = os.path.join(raw_audio_dir, raw_audio_filename)
                    if '{}' in file:
                        file = file.format(len(glob.glob(file.replace('{}', '*'))))
                    
                    write_audio(filename = file, audio = data, rate = self.audio_rate)
                    
                    time_logger.stop_timer('saving audios')

                infos['filename']   = file
                predicted[file]     = infos
                
                time_logger.start_timer('saving json')
                dump_json(map_file, predicted, indent = 4)
                time_logger.stop_timer('saving json')
        
        return results
    