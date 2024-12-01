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
import time
import logging
import numpy as np
import pandas as pd

from tqdm import tqdm

from utils import *
from utils.audio import *
from utils.callbacks import *
from models.utils import prepare_prediction_results
from loggers import timer, time_logger
from utils.keras_utils import TensorSpec, ops
from models.interfaces.base_text_model import BaseTextModel
from models.interfaces.base_audio_model import BaseAudioModel
from utils.text import get_encoder, get_symbols, accent_replacement_matrix, ctc_decode

logger  = logging.getLogger(__name__)

_silent_char    = list(" '\"")
_deep_speech_en_symbols = list(" abcdefghijklmnopqrstuvwxyz'") + ['']

MIN_AUDIO_TIME  = 0.1

DEFAULT_MAX_MEL_LENGTH  = 1024
DEFAULT_MAX_TEXT_LENGTH = min(256, DEFAULT_MAX_MEL_LENGTH // 2)
        
class BaseSTT(BaseTextModel, BaseAudioModel):
    _directories    = {
        ** BaseTextModel._directories, 'search_dir' : '{root}/{self.name}/search'
    }
    
    output_signature    = BaseTextModel.text_signature
    prepare_output  = BaseTextModel.encode_text
    
    def __init__(self,
                 lang,
                 text_encoder   = None,
                 
                 use_fixed_length_input = False,
                 max_input_length   = DEFAULT_MAX_MEL_LENGTH,
                 max_output_length  = DEFAULT_MAX_TEXT_LENGTH,
                 
                 use_ctc_decoder    = None,
                 
                 ** kwargs
                ):
        if text_encoder is None:
            text_encoder = get_encoder(
                text_encoder = text_encoder,
                lang    = lang,
                vocab   = get_symbols(
                    lang, maj = False, arpabet = False, punctuation = 2
                ),
                use_sos_and_eos = self.is_encoder_decoder
            )

        self._init_text(lang = lang, text_encoder = text_encoder, ** kwargs)
        self._init_audio(** kwargs)
        
        self.use_fixed_length_input = use_fixed_length_input
        self.max_input_length   = max_input_length
        self.max_output_length  = max_output_length
        
        super().__init__(** kwargs)
        
        if hasattr(self.model, 'set_tokens'): self.model.set_tokens(** self.model_tokens)

    def build(self, architecture, stt_model = None, ** kwargs):
        if stt_model is None:
            stt_model = {
                'architecture'  : architecture,
                'input_shape'   : self.mel_input_shape,
                'vocab_size'    : self.vocab_size,
                'pad_value'     : self.pad_mel_value,
                'sos_token'     : self.sos_token_idx,
                'eos_token'     : self.eos_token_idx,
                'pad_token'     : self.blank_token_idx,
                ** kwargs
            }

        super(BaseSTT, self).build(stt_model = stt_model)
    
    @property
    def use_ctc_decoder(self):
        return not self.is_encoder_decoder
    
    @property
    def mel_input_shape(self):
        mel_length = self.max_input_length if self.use_fixed_length_input else None
        return (mel_length, ) + self.audio_signature.shape[2:]
    
    @property
    def decoder_method(self):
        return 'greedy' if self.is_encoder_decoder else 'beam'
    
    @property
    def input_signature(self):
        inp_sign = TensorSpec(shape = (None, ) + self.mel_input_shape, dtype = 'float32')
        
        return inp_sign if not self.is_encoder_decoder else (inp_sign, self.text_signature)
    
    @property
    def infer_signature(self):
        return TensorSpec(shape = (None, ) + self.mel_input_shape, dtype = 'float32')

    @property
    def training_hparams(self):
        return super().training_hparams(
            ** self.training_hparams_audio,
            max_input_length    = None,
            max_output_length   = None
        )
    
    def __str__(self):
        des = super().__str__()
        des += self._str_text()
        des += self._str_audio()
        des += "- Use CTC decoder : {}\n".format(self.use_ctc_decoder)
        return des
    
    @timer(name = 'inference', log_if_root = False)
    def infer(self, inputs, training = False, decode = False, prev_tokens = None, ** kwargs):
        output = self.compiled_infer(inputs, training = training, ** kwargs)
        
        return self.decode_output(output) if decode else output
    
    def compile(self, loss = None, metrics = None, loss_config = {}, ** kwargs):
        if loss is None:
            loss = 'TextLoss' if self.is_encoder_decoder else 'CTCLoss'
        if metrics is None:
            if self.is_encoder_decoder:
                metrics = ['TextAccuracy']
            else:
                metrics = [{'metric' : 'TextMetric', 'config': {'pad_value' : self.blank_token_idx}}]
        
        loss_config['pad_value'] = self.blank_token_idx
        super().compile(loss = loss, metrics = metrics, loss_config = loss_config, ** kwargs)
    
    def decode_output(self, output, * args, ** kwargs):
        if self.use_ctc_decoder:
            return self.text_encoder.ctc_decode(output, ** kwargs)
        return self.decode_text(output, ** kwargs)
    
    @timer
    def distance(self, hypothesis, truth, ** kwargs):
        kwargs.setdefault('insertion_cost', {})
        kwargs.setdefault('deletion_cost', {})
        kwargs.setdefault('replacement_cost', {})
        
        for c in _silent_char + [self.sep_token]:
            kwargs['insertion_cost'].setdefault(c, 0)
            kwargs['deletion_cost'].setdefault(c, 0)
        
        kwargs['replacement_cost'] = {
            ** accent_replacement_matrix, ** kwargs['replacement_cost']
        }
        
        return self.text_encoder.distance(hypothesis, truth, ** kwargs)
    
    def prepare_input(self, data, pad_or_trim = True):
        audio = self.get_audio(data)
        
        if pad_or_trim:
            audio = self.pad_or_trim(audio)
        
        return audio
    
    def pad_or_trim(self, audio):
        if ops.shape(audio)[0] > self.max_input_length:
            audio = audio[: self.max_input_length]
        elif self.use_fixed_length_input and ops.shape(audio)[0] != self.max_input_length:
            audio = ops.pad(
                audio, [(0, self.max_input_length - ops.shape(audio)[0]), (0, 0)],
                constant_values = self.pad_mel_value
            )
        
        return audio
    
    def prepare_data(self, data):
        mel  = self.prepare_input(data)
        text = self.prepare_output(data)
        
        if not self.is_encoder_decoder: return mel, text
        
        text_in, text_out = text_in[:-1], text_out[1:]
        
        return (mel, text_in), text_out
    
    def filter_data(self, inputs, output):
        mel = inputs[0] if isinstance(inputs, tuple) else inputs
        return ops.logical_and(
            ops.shape(mel)[0] <= self.max_input_length,
            ops.shape(output)[0] <= self.max_output_length
        )
        
    def augment_data(self, inputs, output):
        if not isinstance(inputs, tuple):
            return self.augment_audio(inputs, max_length = self.max_input_length), output

        mel = self.augment_audio(inputs[0], max_length = self.max_input_length)
        return (mel, ) + inputs[1:], output
        
    def get_dataset_config(self, ** kwargs):
        pad_values  = (self.pad_mel_value, self.blank_token_idx)
        if self.is_encoder_decoder:
            pad_values = (pad_values, self.blank_token_idx)
        
        kwargs.update({
            'pad_kwargs'    : {'padding_values' : pad_values}
        })
        if self.use_fixed_length_input:
            pad_shapes  = (self.mel_input_shape, (None, ))
            if self.is_encoder_decoder:
                pad_shapes = (pad_values, (None, ))
            kwargs['pad_kwargs']['padded_shapes'] = pad_shapes
        
        return super().get_dataset_config(** kwargs)
    
    def get_prediction_callbacks(self,
                                 *,

                                 save    = True,
                                 
                                 directory  = None,
                                 raw_audio_dir  = None,
                                 
                                 filename   = 'audio_{}.mp3',
                                 # Verbosity config
                                 verbose = 1,
                                 
                                 post_processing    = None,
                                 
                                 use_multithreading = False,

                                 ** kwargs
                                ):
        """
            Return a list of `utils.callbacks.Callback` instances that handle data saving/display
            
            Arguments :
                - save  : whether to save detection results
                          Set to `True` if `save_boxes` or `save_detected` is True
                - save_empty    : whether to save raw images if no object has been detected
                - save_detected : whether to save the image with detected objects
                - save_boxes    : whether to save boxes as individual images (not supported yet)
                
                - directory : root directory for saving (see below for the complete tree)
                - raw_img_dir   : where to save raw images (default `{directory}/images`)
                - detected_dir  : where to save images with detection (default `{directory}/detected`)
                - boxes_dir     : where to save individual boxes (not supported yet)
                
                - filename  : raw image file format
                - detected_filename : image with detection file format
                - boxes_filename    : individual boxes file format
                
                - display   : whether to display image with detection
                              If `None`, set to `True` if `save == False`
                - verbose   : verbosity level (cumulative, i.e., level 2 includes level 1)
                              - 1 : displays the image with detection
                              - 2 : displays the individual boxes
                              - 3 : logs the boxes position
                                 
                - post_processing   : callback function applied on the results
                                      Takes as input all kwargs returned by `self.predict`
                                      - image   : the raw original image (`ndarray / Tensor`)
                                      - boxes   : the detected objects (`dict`)
                                      * filename    : the image file (`str`)
                                      * detected    : the image with detection (`ndarray`)
                                      * output      : raw model output (`Tensor`)
                                      * frame_index : the frame index in a stream (`int`)
                                      Entries with "*" are conditionned and are not always provided
                
                - use_multithreading    : whether to multi-thread the saving callbacks
                
                - kwargs    : mainly ignored
            Return : (predicted, required_keys, callbacks)
                - predicted : the mapping `{filename : infos}` stored in `{directory}/map.json`
                - required_keys : expected keys to save (see `models.utils.should_predict`)
                - callbacks : the list of `Callback` to be applied on each prediction
        """
        if save is None:    save = not verbose
        if verbose is None: verbose = not save
        
        if directory is None: directory = self.pred_dir
        map_file    = os.path.join(directory, 'map.json')
        
        predicted   = {}
        callbacks   = []
        required_keys   = ['text']
        if save:
            predicted   = load_json(map_file, {})
            
            required_keys.append('filename')
            if raw_audio_dir is None: raw_audio_dir = os.path.join(directory, 'audios')
            callbacks.append(AudioSaver(
                key = 'filename',
                name    = 'saving raw',
                data_key    = 'audio',
                file_format = os.path.join(raw_audio_dir, filename),
                index_key   = 'segment_index',
                use_multithreading  = use_multithreading
            ))
        
            callbacks.append(JSonSaver(
                data    = predicted,
                filename    = map_file,
                force_keys  = {'alignment'},
                primary_key = 'filename',
                use_multithreading = use_multithreading
            ))
        
        if verbose:
            callbacks.append(AudioDisplayer(show_text = True))
        
        if post_processing is not None:
            callbacks.append(FunctionCallback(post_processing))
        
        return predicted, required_keys, callbacks


    def _add_segment(self, all_segments, segment, start, end, tokens, result, lang = None, ** kwargs):
        text    = self.decode_output(tokens) if len(tokens) > 0 else ''
        if isinstance(text, list): text = text[0]
        text    = text.strip()

        infos = {
            "num"   : len(all_segments),
            "start" : start,
            "end"   : end,
            "time"  : end - start,
            "text"  : text,
            "tokens"    : tokens,
            "score"     : result.score[0].numpy() if hasattr(result, 'score') else 0
        }
        if lang: infos['lang'] = lang
        if text: all_segments.append(infos)

        return infos

    def predict_segment(self, mel, start, end, time_window = 30, ** kwargs):
        window_sample   = self._get_sample_index(time_window)
        start_sample    = self._get_sample_index(start)
        if end:
            end_sample  = self._get_sample_index(end)
        else:
            end, end_sample = self._get_sample_time(len(mel)), len(mel)
        
        segments = []
        for i, start in enumerate(range(start_sample, end_sample, window_sample)):
            segment = mel[start : min(end_sample, start + window_sample)]
            pred = self.infer(
                ops.expand_dims(segment, axis = 0), decode = False, ** kwargs
            )

            segment_infos = self._add_segment(
                segments,
                segment = segment,
                start   = start + time_window * i,
                end     = min(end, start + time_window * (i + 1)),
                tokens  = pred if not hasattr(pred, 'tokens') else pred.tokens,
                result  = pred,
                ** kwargs
            )
            
            if part_post_processing is not None:
                part_post_processing(segment_infos, segment = segment)
        
        return segments
    
    @timer
    def predict(self,
                audios,
                alignments  = None,
                *,
                
                verbose = True,
                overwrite   = False,

                condition_on_previous_text  = True,
                
                part_post_processing    = None,
                
                predicted   = None,
                _callbacks  = None,
                required_keys   = None,
                
                ** kwargs
               ):
        ####################
        #  Initialization  #
        ####################
        
        if alignments is not None and not isinstance(alignments, list): alignments = [alignments]
            
        now = time.time()
        with time_logger.timer('initialization'):
            join_callbacks = _callbacks is None
            if _callbacks is None:
                predicted, required_keys, _callbacks = self.get_prediction_callbacks(
                    verbose = verbose, ** kwargs
                )
        
            results, inputs, indexes, files, duplicates, filtered = prepare_prediction_results(
                audios,
                predicted,
                
                rank    = 1,
                primary_key = 'filename',
                expand_files    = True,
                normalize_entry = path_to_unix,
                
                overwrite   = overwrite,
                required_keys   = required_keys,
            )
        
        show_idx = apply_callbacks(results, 0, _callbacks)
        
        for idx, file, data in zip(indexes, files, inputs):
            if isinstance(file, str) and verbose:
                logger.info('Processing file {}...'.format(file))

            with time_logger.timer('loading audio'):
                mel = self.get_input(data, pad_or_trim = False)

            # Get associated alignment (if provided)
            alignment = alignments[idx] if alignments is not None else None
            
            if alignments:
                alignment = alignments[idx]
            elif isinstance(data, AudioAnnotation):
                alignment = data._alignment
            elif isinstance(data, dict) and 'alignment' in data:
                alignment = data['alignment']
            else:
                alignment = [{'id' : -1, 'start' : 0., 'end' : None}]
            
            all_segments = []
            for align in alignment:
                segments = self.predict_segment(
                    mel,
                    verbose = verbose,
                    part_post_processing    = part_post_processing,
                    ** {** kwargs, ** align}
                )
                if isinstance(segments, dict): segments = [segments]
                all_segments.extend([
                    {** align, ** segment} for segment in segments
                ])
            
            infos = {} if not isinstance(data, dict) else data.copy()
            infos.update({
                'text'  : ' '.join([seg['text'] for seg in all_segments]),
                'alignment' : all_segments
            })
            if not isinstance(data, dict):
                if file: infos['filename'] = file
                infos['audio'] = data
            
            if file:
                for idx in duplicates[file]:
                    results[idx] = (predicted.get(file, {}), infos)
            else:
                results[idx] = ({}, infos)
            
            show_idx = apply_callbacks(results, show_idx, _callbacks)
            
        
        return results
    
    @timer
    def stream(self, max_time = 30, filename = None, ** kwargs):
        try:
            import sounddevice as sd
        except ImportError as e:
            logger.error('You must install `sounddevice` : `pip install sounddevice`')
            return None
        
        t0 = time.time()
        audio = sd.rec(
            samplerate  = self.audio_rate, 
            out         = np.zeros((int(self.audio_rate * max_time), 1))
        )
        
        input("Recording... Press enter to stop")
        sd.stop()
        t1 = time.time() - t0

        audio = np.reshape(audio, [-1])[2000:int(t1 * self.audio_rate)]
        audio = audio / np.max(np.abs(audio))

        text = self.infer(self.get_input(audio, pad_or_trim = False), decode = True, ** kwargs)[0]
        
        _ = display_audio(audio, rate = self.audio_rate)

        print("\n\nPrediction : {}\n".format(text))
        
        if filename is not None:
            write_audio(audio = audio, filename = filename, rate = self.audio_rate)
    
    @timer
    def search(self, keyword, audios, threshold = 0.8, ** kwargs):
        # Get predictions for audios
        pred = self.predict(audios, ** kwargs)

        return SearchResult(* [AudioSearch(
            keyword = keyword, distance_fn = self.distance, rate = self.audio_rate,
            directory = self.search_dir, filename = infos['filename'],
            infos = infos['alignment'], threshold = threshold
        ) for _, infos in pred])
    
    def get_config(self, * args, ** kwargs):
        config = super().get_config(* args, ** kwargs)    
        config.update({
            ** self.get_config_audio(),
            ** self.get_config_text(),
            
            'use_fixed_length_input'    : self.use_fixed_length_input,
            'max_input_length'  : self.max_input_length,
            'max_output_length' : self.max_output_length
        })
        
        return config

