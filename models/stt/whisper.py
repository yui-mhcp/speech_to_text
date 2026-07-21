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
import contextlib
import numpy as np

from tqdm import tqdm
from functools import cached_property

from .base_stt import BaseSTT
from loggers import Timer, timer
from utils.text import process_model_output, mask_tokens, mask_batch_tokens
from utils.keras import ops, graph_compile

logger  = logging.getLogger(__name__)

class Whisper(BaseSTT):
    def __init__(self, lang = 'multi', pretrained = 'openai/whisper-base', ** kwargs):
        if pretrained:
            kwargs.update({
                'lang'  : 'en' if 'en' in pretrained else 'multi',
                'tokenizer' : pretrained,
                
                'rate'  : 16000,
                'mel_fn'    : 'WhisperSTFT',
                'mel_config'    : {'n_mel_channels' : 128 if 'large' in pretrained else 80},
                'load_audio_kwargs' : {
                    'normalize'     : 32768.,
                    'read_method'   : 'read_ffmpeg'
                },
                
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
    def notimestamp_token(self):
        return '<|notimestamps|>'
    
    @property
    def timestamp_begin_idx(self):
        return self.notimestamp_token_idx + 1
    
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
    
    @property
    def notimestamp_token_idx(self):
        return self.tokenizer[self.notimestamp_token]

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

    def pad_or_trim(self, audio):
        if ops.shape(audio)[0] > self.max_input_length:
            audio = audio[: self.max_input_length]
        elif self.use_fixed_length_input and ops.shape(audio)[0] != self.max_input_length:
            # `WhisperSTFT` clamps the log-mel at `max - 8` then rescales by `(x + 4) / 4`,
            # placing silence at `max - 2` in the normalized space : pad with silence, as
            # openai/whisper does by zero-padding the audio (not with `pad_mel_value`)
            audio = ops.pad(
                audio, [(0, self.max_input_length - ops.shape(audio)[0]), (0, 0)],
                constant_values = ops.max(audio) - 2.
            )

        return audio

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
        """
            Language detection via the keras architecture (`self.model.{encoder / decoder}`).
            Not supported by the TRT-LLM runtime : leave `lang = None` at inference instead,
            the language is then inferred from the generated tokens (see `_infer_segments`).
        """
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
                        input_tokens  = None,
                        verbose = True,
                        
                        force_detect_language   = False,
                        condition_on_previous_text = True,
                        
                        segment_processing  = None,
                        
                        ** kwargs
                       ):
        kwargs['encoder_output_lengths']    = self.max_input_length // 2
        kwargs.setdefault('max_length', self.max_output_length)
        # `max_length` is the keras-runtime argument ; TRT-LLM expects `max_new_tokens`
        # (which otherwise defaults to 1 in `CustomModelRunnerCpp.generate` !)
        kwargs.setdefault('max_new_tokens', kwargs['max_length'])

        seek    = kwargs.pop('seek', 0)
        n_frames    = len(mel)
        prev_seek   = seek
        input_stride    = 2 # 3000 // 1500

        logits_processors   = kwargs.pop('logits_processors', None)

        all_tokens, segments = [], []
        with tqdm(total = n_frames, unit = 'frames', disable = verbose == 0) as pbar:
            while seek < n_frames:
                with Timer('segment processing'):
                    segment = mel[seek : seek + self.max_input_length]
                    segment_length = len(segment)
                    segment = self.pad_or_trim(segment)

                    if lang is None and force_detect_language:
                        lang, _ = self.detect_language(segment)

                    if input_tokens is None:
                        inputs = self.get_inference_tokens(lang = lang, ** kwargs)
                    else:
                        inputs = input_tokens

                    if condition_on_previous_text and len(all_tokens):
                        n_prev = kwargs['max_length'] // 2 - 1
                        if self.runtime == 'trt_llm':
                            # the TRT-LLM decoder is compiled with a maximum prompt length
                            n_prev = min(n_prev, self.model.max_input_length - len(inputs) - 1)

                        if n_prev > 0:
                            inputs = np.array(
                                [self.start_of_prev_token_idx] +
                                list(all_tokens[- n_prev :]) +
                                list(inputs), 'int32'
                            )
                        else:
                            inputs = np.array(inputs, 'int32')
                    else:
                        inputs = np.array(inputs, 'int32')

                    processor = logits_processors
                    if processor is None and self.runtime in ('trt_llm', 'trt_llm_api'):
                        # enforces the whisper timestamp rules at generation time
                        # (stateful : a new instance is required for each window)
                        # when the prompt is `[sos_token]` (`lang = None`), the language and
                        # task tokens are generated first : skip them in the processor
                        processor = WhisperTimestampLogitsProcessor(
                            self, sample_begin = 2 if len(inputs) == 1 else 0
                        )

                infer_kwargs = kwargs
                if processor is not None:
                    infer_kwargs = {** kwargs, 'logits_processors' : processor}

                tokens = self.compiled_infer(
                    segment[None], tokens = inputs[None], tokens_length = len(inputs),
                    ** infer_kwargs
                )
                if hasattr(tokens, 'tokens'):
                    tokens = process_model_output(tokens)[0]
                else:
                    tokens = tokens[0]
                
                if tokens and isinstance(tokens[0], list):
                    tokens = tokens[0]
                tokens = np.array(tokens, dtype = np.int32)

                # the timestamp detection below assumes the sequence ends with a timestamp
                # (or text), not with the EOS token
                if len(tokens) and tokens[-1] == self.eos_token_idx:
                    tokens = tokens[:-1]

                if lang is None and len(tokens) >= 2:
                    lang    = self._idx_to_lang.get(tokens[0], None)
                    tokens  = tokens[2:]
                
                with Timer('post_processing'):
                    timestamp_offset = self._get_sample_time(seek)

                    timestamp_tokens    = tokens >= self.timestamp_begin_idx
                    # a single trailing timestamp (`... text <|t|>`) means there is no speech
                    # after it : the window is fully transcribed
                    single_timestamp_ending = (
                        len(tokens) >= 2 and timestamp_tokens[-1] and not timestamp_tokens[-2]
                    )
                    consecutive         = np.where(np.logical_and(
                        timestamp_tokens[:-1], timestamp_tokens[1:]
                    ))[0] + 1
                    # if the output contains two consecutive timestamp tokens
                    if len(consecutive) > 0:
                        slices = consecutive.tolist()
                        if single_timestamp_ending:
                            # the trailing `<|start|> text <|end|>` sub-segment is complete :
                            # treat it as a regular slice instead of re-decoding it next window
                            slices.append(len(tokens))

                        last_slice = 0
                        for current_slice in slices:
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

                        if single_timestamp_ending:
                            # no speech after the last timestamp : skip to the next window
                            seek += segment_length
                        else:
                            last_timestamp_position = (
                                tokens[last_slice - 1] - self.timestamp_begin_idx
                            )
                            advance = int(last_timestamp_position) * input_stride
                            if advance <= 0:
                                # malformed timestamps (should not happen when the timestamp
                                # rules are enforced) : never seek backwards, skip the window
                                logger.warning(
                                    'Inconsistent end timestamp for the window at frame {} '
                                    '(advance of {} frames) : skipping to the next window'.format(
                                        seek, advance
                                    )
                                )
                                advance = segment_length
                            seek += advance
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

                        # advance by the effective (un-padded) window length
                        seek += segment_length
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

class WhisperTimestampLogitsProcessor:
    """
        TRT-LLM `logits_post_processor` enforcing the whisper timestamp rules (the TRT-LLM
        counterpart of the keras-runtime `timestamp_filter` below) :
            1) The generation must start with a timestamp (at most `max_initial_timestamp`)
            2) Timestamps come in pairs : after a segment-start timestamp, only its end
               timestamp (or EOS) is allowed, and after a complete pair, text is expected
            3) Timestamps are monotonically increasing
            4) If the cumulated probability of timestamps exceeds every text token, a
               timestamp is sampled

        Without these rules, the model may output malformed / non-monotonic timestamps,
        making the `seek` update in `_infer_segments` unreliable for audios longer than
        one window (30 sec).

        The processor is stateful (`_step`) : create a new instance for each `generate` call.
    """
    def __init__(self, model, sample_begin = 0, max_initial_timestamp = 1.):
        self.timestamp_begin    = int(model.timestamp_begin_idx)
        self.eos_token  = int(model.eos_token_idx)
        self.remove_tokens  = model.remove_tokens
        self.remove_tokens_at_start = model.remove_tokens_with_space
        # number of generated tokens to skip before applying the rules : when the prompt is
        # reduced to `[sos_token]`, the language and task tokens are generated (not forced)
        self.sample_begin   = sample_begin
        self.max_initial_timestamp_index    = round(
            max_initial_timestamp / model.time_precision
        )

        self._step  = 0
        self._remove_indexes    = None
        self._remove_indexes_at_start   = None

    def __call__(self, req_id, logits, ids, stream_ptr, client_id):
        import torch

        if self._step < self.sample_begin:
            # the model is generating the language / task tokens : leave them unconstrained
            self._step += 1
            return

        # `stream_ptr` is None with the pytorch-backend LLM API : the processor already
        # runs on the generation stream
        stream = (
            torch.cuda.stream(torch.cuda.ExternalStream(stream_ptr))
            if stream_ptr is not None else contextlib.nullcontext()
        )
        with stream:
            if self._remove_indexes is None:
                self._remove_indexes    = torch.as_tensor(
                    self.remove_tokens, dtype = torch.long, device = logits.device
                )
                self._remove_indexes_at_start   = torch.as_tensor(
                    self.remove_tokens_at_start, dtype = torch.long, device = logits.device
                )

            # (num_beams, vocab_size) view, whatever the actual layout (e.g. (1, beams, vocab))
            scores = logits.view(-1, logits.shape[-1])

            if self._step == self.sample_begin:
                scores[:, self._remove_indexes_at_start] = float('-inf')
                # the generation must start with a timestamp, at most `max_initial_timestamp`
                scores[:, : self.timestamp_begin] = float('-inf')
                scores[:, self.timestamp_begin + self.max_initial_timestamp_index + 1 :] = float('-inf')
            else:
                scores[:, self._remove_indexes] = float('-inf')

                for k in range(scores.shape[0]):
                    # the last `_step - sample_begin` tokens are the transcription ones
                    # (`ids` may or may not include the prompt tokens)
                    seq = [int(t) for t in ids[k][- (self._step - self.sample_begin) :]]
                    last_was_timestamp        = seq[-1] >= self.timestamp_begin
                    penultimate_was_timestamp = len(seq) < 2 or seq[-2] >= self.timestamp_begin

                    if last_was_timestamp:
                        if penultimate_was_timestamp:
                            # a (start, end) pair is complete : expect text
                            scores[k, self.timestamp_begin :] = float('-inf')
                        else:
                            # a segment started : expect its end timestamp (or EOS)
                            scores[k, : self.eos_token] = float('-inf')

                    timestamps = [t for t in seq if t >= self.timestamp_begin]
                    if timestamps:
                        # timestamps must not decrease (but a segment end may equal its start)
                        last = timestamps[-1]
                        if not last_was_timestamp or penultimate_was_timestamp:
                            last += 1
                        scores[k, self.timestamp_begin : last] = float('-inf')

                # if the cumulated probability of timestamps exceeds every text token,
                # force sampling a timestamp
                logprobs    = torch.log_softmax(scores.float(), dim = -1)
                timestamp_logprob   = logprobs[:, self.timestamp_begin :].logsumexp(dim = -1)
                max_text_logprob    = logprobs[:, : self.timestamp_begin].max(dim = -1).values
                scores[timestamp_logprob > max_text_logprob, : self.timestamp_begin] = float('-inf')

        self._step += 1

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