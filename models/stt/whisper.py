# Copyright (C) 2025-now yui-mhcp project author. All rights reserved.
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

from tqdm import tqdm
from functools import cached_property

from .base_stt import BaseSTT
from loggers import Timer, timer
from utils.text import process_model_output, mask_tokens, mask_batch_tokens
from utils.keras import ops, graph_compile

class Whisper(BaseSTT):
    def __init__(self, lang = 'multi', pretrained = 'base', ** kwargs):
        if pretrained:
            self.trim_kwargs = {'normalize' : 32768.}
            
            kwargs.update({
                'lang'  : 'en' if 'en' in pretrained else 'multi',
                'tokenizer' : 'whisper',
                
                'rate'  : 16000,
                'mel_fn'    : 'WhisperSTFT',
                'mel_config'    : {'n_mel_channels' : 128 if 'large' in pretrained else 80},
                
                'max_input_length'  : 3000,
                'use_fixed_length_input'    : True,
                
                'pretrained_name'   : pretrained
            })
        else:
            kwargs['lang'] = lang
        
        kwargs.update({
            'audio_format'  : 'mel',
            'architecture'  : 'Whisper'
        })
        super().__init__(pretrained = pretrained, ** kwargs)
        
        self.trim_kwargs['read_method'] = 'read_ffmpeg'
        
        self._lang_to_idx   = {
            v.strip('<|>') : i for i, v in enumerate(self.vocab)
            if v.startswith('<|') and v.strip('<|>') in LANGUAGES
        }
        self._idx_to_lang   = {i : v for v, i in self._lang_to_idx.items()}
        
        t = list(self.non_speech_token_indexes) + list(self.special_token_indexes)

        self.remove_tokens              = np.array(t, 'int32')
        self.remove_tokens_with_space   = np.array([
            self.tokenizer[' '], self.eos_token_idx
        ] + t, 'int32')

    def build(self, *, pretrained = None, model = None, stt_model = None, ** kwargs):
        if stt_model is not None: model = stt_model
        elif model is None:
            if pretrained:
                from architectures.transformers import whisper_arch
                
                model = whisper_arch.Whisper.from_pretrained(
                    pretrained = pretrained, decoder_eos_token = self.eos_token_idx, ** kwargs
                )
            else:
                model = kwargs
        
        super().build(model = model)
    
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
        return '<|nospeech|>' if '<|nospeech|>' in self.tokenizer else '<|nocaptions|>'
    
    @property
    def timestamp_begin_idx(self):
        return self.vocab_size
    
    @cached_property
    def languages(self):
        return list(self._lang_to_idx.keys())

    @property
    def sos_token_idx(self):
        return self.tokenizer[self.sos_token]
    
    @property
    def transcribe_token_idx(self):
        return self.tokenizer[self.transcribe_token]
        
    @property
    def translate_token_idx(self):
        return self.tokenizer[self.translate_token]
    
    @property
    def start_of_prev_token_idx(self):
        return self.tokenizer[self.start_of_prev_token]
    
    @property
    def nospeech_token_idx(self):
        return self.tokenizer[self.nospeech_token]
    
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
        result = {self.tokenizer[" -"], self.tokenizer[" '"]}
        for symbol in symbols + list(miscellaneous):
            for tokens in [self.tokenizer[symbol], self.tokenizer[" " + symbol]]:
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
        return [self.tokenizer[token] for token in self.special_tokens]
    
    @cached_property
    def segment_duration(self):
        return self._get_sample_time(self.max_input_length)
    
    @cached_property
    def time_precision(self):
        return self._get_sample_time(2)
    
    @graph_compile(prefer_xla = True)
    def compiled_detect_language(self, mel, tokens = None, training = False):
        encoder_output = self.model.encoder(mel, training = training)
        
        if tokens is None:
            tokens = ops.fill((1, 1), self.sos_token_idx)
        
        pred = self.model.decoder(
            tokens, encoder_output = encoder_output, training = training
        )
        return ops.softmax(ops.take(
            pred[0, -1, :], ops.convert_to_tensor(self.language_indexes, 'int32')
        ), axis = -1)
    
    @timer
    def detect_language(self, audio, ** kwargs):
        with Timer('pre_processing'):
            mel = self.get_input(audio, pad_or_trim = True)
            if ops.rank(mel) == 2: mel = ops.expand_dims(mel, axis = 0)

            tokens  = ops.fill((mel.shape[0], 1), self.sos_token_idx)

        probs   = self.compiled_detect_language(mel = mel, tokens = tokens, ** kwargs)
        probs   = ops.convert_to_numpy(probs)
        
        return (
            self.languages[np.argmax(probs)],
            {lang : p for lang, p in zip(self.languages, probs)}
        )

    def _infer_segments(self,
                        mel,
                        *,
                        
                        lang    = None,
                        verbose = True,
                        
                        force_detect_language   = False,
                        condition_on_previous_text = True,
                        
                        segment_processing  = None,
                        
                        ** kwargs
                       ):
        kwargs['encoder_output_lengths']    = 1500
        kwargs.setdefault('max_length', self.max_output_length)
        
        seek    = kwargs.pop('seek', 0)
        n_frames    = len(mel)        
        prev_seek   = seek
        input_stride    = 2 # 3000 // 1500

        all_tokens, segments = [], []
        with tqdm(total = n_frames, unit = 'frames', disable = verbose == 0) as pbar:
            while seek < n_frames:
                with Timer('segment processing'):
                    segment = mel[seek : seek + self.max_input_length]
                    segment_length = len(segment)
                    segment = self.pad_or_trim(segment)

                    if lang is None and force_detect_language:
                        lang, _ = self.detect_language(segment)

                    tokens = self.get_inference_tokens(lang = lang, ** kwargs)
                    if condition_on_previous_text and len(all_tokens):
                        tokens = np.array(
                            [self.start_of_prev_token_idx] +
                            all_tokens[- (kwargs['max_length'] // 2 - 1) :] +
                            tokens, 'int32'
                        )
                    else:
                        tokens = np.array(tokens, 'int32')

                tokens = self.compiled_infer(
                    segment[None], tokens = tokens[None], tokens_length = len(tokens), ** kwargs
                )
                if hasattr(tokens, 'tokens'):
                    tokens = process_model_output(tokens)[0]
                else:
                    tokens = tokens[0]
                
                if tokens and isinstance(tokens[0], list):
                    tokens = tokens[0]
                tokens = np.array(tokens, dtype = np.int32)
                
                if lang is None:
                    lang    = self._idx_to_lang.get(tokens[0], None)
                    tokens  = tokens[2:]
                
                with Timer('post_processing'):
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
                            
                            segments.append({
                                "start" : timestamp_offset + start_timestamp_position * self.time_precision,
                                "end"   : timestamp_offset + end_timestamp_position * self.time_precision,
                                "text"  : self.decode_output(sliced_tokens),
                                "tokens"    : sliced_tokens[sliced_tokens < self.eos_token_idx],
                                'lang'  : lang
                            })
                            
                            if segment_processing is not None:
                                segment_processing(segments[-1], segment = segment)
                            
                            last_slice = current_slice

                        last_timestamp_position = (
                            tokens[last_slice - 1] - self.timestamp_begin_idx
                        )
                        seek += last_timestamp_position * input_stride
                        all_tokens.extend(tokens[: last_slice + 1])
                    else:
                        duration    = self._get_sample_time(segment_length)
                        timestamps  = tokens[timestamp_tokens]
                        if len(timestamps) > 0 and timestamps[-1] != self.timestamp_begin_idx:
                            # no consecutive timestamps but it has a timestamp; use the last one.
                            # single timestamp at the end means no speech after the last timestamp.
                            last_timestamp_position = timestamps[-1] - self.timestamp_begin_idx
                            duration = float(last_timestamp_position) * self.time_precision

                        tokens = tokens[tokens < self.eos_token_idx]
                        segments.append({
                            "start" : timestamp_offset,
                            "end"   : timestamp_offset + duration,
                            "text"  : self.decode_output(tokens),
                            "tokens"    : tokens,
                            'lang'  : lang
                        })
                        
                        if segment_processing is not None:
                            segment_processing(segments[-1], segment = segment)

                        seek += len(segment)
                        all_tokens.extend(tokens)

                    # update progress bar
                    pbar.update(min(n_frames, seek) - prev_seek)
                    prev_seek = seek

        for segment in segments: segment['time'] = segment['end'] - segment['start']
        
        return segments

    def get_inference_tokens(self, lang = None, task = None, ** _):
        return [
            self.sos_token_idx,
            self._lang_to_idx[lang],
            self.translate_token_idx if task == 'translate' else self.transcribe_token_idx
        ] if lang else [self.sos_token_idx]

def add_batch_index(indices, batch_size, mask = None):
    if mask is None:
        batch_indexes = ops.arange(batch_size)
    else:
        indexes = ops.where(mask)
        indexes = indexes[0] if isinstance(indexes, list) else indexes[:, 0]
        batch_indexes = ops.convert_to_tensor(indexes, 'int32')
    
    indices = ops.convert_to_tensor(indices, 'int32')
    return ops.stack([
        ops.repeat(batch_indexes, ops.shape(indices)[0]),
        ops.tile(indices, [ops.shape(batch_indexes)[0]])
    ], axis = 1)

def timestamp_filter(self, scores, tokens, to_remove, state, max_initial_timestamp = 1, ** _):
    if state.state is None:
        # suppress generating non-timestamp tokens at the beginning
        to_remove = ops.concat([
            to_remove, ops.arange(self.timestamp_begin_idx)
        ], axis = -1)

        # apply the `max_initial_timestamp` option
        if max_initial_timestamp > 0:
            to_remove = ops.concat([
                to_remove, ops.range(self.timestamp_begin_idx + max_initial_timestamp, ops.shape(scores)[-1])
            ], axis = -1)

        scores = mask_batch_tokens(scores, to_remove)
    else:
        batch_size = ops.shape(scores)[0]

        last_was_timestamp          = tokens[:, -1] >= self.timestamp_begin_idx
        penultimate_was_timestamp   = ops.cond(
            state.t < 2,
            lambda: ops.ones((ops.shape(tokens)[0], ), dtype = 'bool'),
            lambda: tokens[:, -2] >= self.timestamp_begin_idx
        )

        to_remove_batch = add_batch_index(to_remove, batch_size)

        if ops.any(last_was_timestamp):
            last_but_not_penultimate    = ops.logical_and(
                last_was_timestamp, ops.logical_not(penultimate_was_timestamp)
            )
            last_and_penultimate    = ops.logical_and(
                last_was_timestamp, penultimate_was_timestamp
            )
            if ops.any(last_but_not_penultimate):
                to_remove_batch = ops.concat([
                    to_remove_batch,
                    add_batch_index(ops.range(self.eos_token_idx), batch_size, last_but_not_penultimate)
                ], axis = 0)

            if ops.any(last_and_penultimate):
                to_remove_batch = ops.concat([
                    to_remove_batch,
                    add_batch_index(ops.range(self.timestamp_begin_idx, ops.shape(scores)[-1]), batch_size, last_and_penultimate)
                ], axis = 0)

        scores = mask_tokens(scores, to_remove_batch)
        # if sum of probability over timestamps is above any other token, sample timestamp
        logits  = ops.log_softmax(scores)

        timestamp_logits = ops.logsumexp(logits[:, self.timestamp_begin_idx :], axis = -1)
        max_text_logits  = ops.max(logits[:, : self.timestamp_begin_idx], axis = -1)

        timestamp_over_text = timestamp_logits > max_text_logits
        if ops.any(timestamp_over_text):
            scores = mask_tokens(
                scores, add_batch_index(ops.range(self.timestamp_begin_idx), batch_size, timestamp_over_text)
            )

    return scores

def logits_filter(self, scores, tokens, state, ** _):
    to_remove = self.remove_tokens_with_space if state.state is None else self.remove_tokens
    return timestamp_filter(self, scores, tokens[:, :state.t], to_remove, state)

LANGUAGES = {
    "en": "english",
    "zh": "chinese",
    "de": "german",
    "es": "spanish",
    "ru": "russian",
    "ko": "korean",
    "fr": "french",
    "ja": "japanese",
    "pt": "portuguese",
    "tr": "turkish",
    "pl": "polish",
    "ca": "catalan",
    "nl": "dutch",
    "ar": "arabic",
    "sv": "swedish",
    "it": "italian",
    "id": "indonesian",
    "hi": "hindi",
    "fi": "finnish",
    "vi": "vietnamese",
    "he": "hebrew",
    "uk": "ukrainian",
    "el": "greek",
    "ms": "malay",
    "cs": "czech",
    "ro": "romanian",
    "da": "danish",
    "hu": "hungarian",
    "ta": "tamil",
    "no": "norwegian",
    "th": "thai",
    "ur": "urdu",
    "hr": "croatian",
    "bg": "bulgarian",
    "lt": "lithuanian",
    "la": "latin",
    "mi": "maori",
    "ml": "malayalam",
    "cy": "welsh",
    "sk": "slovak",
    "te": "telugu",
    "fa": "persian",
    "lv": "latvian",
    "bn": "bengali",
    "sr": "serbian",
    "az": "azerbaijani",
    "sl": "slovenian",
    "kn": "kannada",
    "et": "estonian",
    "mk": "macedonian",
    "br": "breton",
    "eu": "basque",
    "is": "icelandic",
    "hy": "armenian",
    "ne": "nepali",
    "mn": "mongolian",
    "bs": "bosnian",
    "kk": "kazakh",
    "sq": "albanian",
    "sw": "swahili",
    "gl": "galician",
    "mr": "marathi",
    "pa": "punjabi",
    "si": "sinhala",
    "km": "khmer",
    "sn": "shona",
    "yo": "yoruba",
    "so": "somali",
    "af": "afrikaans",
    "oc": "occitan",
    "ka": "georgian",
    "be": "belarusian",
    "tg": "tajik",
    "sd": "sindhi",
    "gu": "gujarati",
    "am": "amharic",
    "yi": "yiddish",
    "lo": "lao",
    "uz": "uzbek",
    "fo": "faroese",
    "ht": "haitian creole",
    "ps": "pashto",
    "tk": "turkmen",
    "nn": "nynorsk",
    "mt": "maltese",
    "sa": "sanskrit",
    "lb": "luxembourgish",
    "my": "myanmar",
    "bo": "tibetan",
    "tl": "tagalog",
    "mg": "malagasy",
    "as": "assamese",
    "tt": "tatar",
    "haw": "hawaiian",
    "ln": "lingala",
    "ha": "hausa",
    "ba": "bashkir",
    "jw": "javanese",
    "su": "sundanese",
}
