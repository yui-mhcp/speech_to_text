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
import time
import logging
import numpy as np
import pandas as pd

from tqdm import tqdm

from utils import *
from utils.audio import *
from utils.callbacks import *
from loggers import Timer, timer
from utils.keras import TensorSpec, ops
from ..interfaces.base_text_model import BaseTextModel
from ..interfaces.base_audio_model import BaseAudioModel
from utils.text import get_tokenizer, get_symbols

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
                 *,
                 
                 use_fixed_length_input = False,
                 max_input_length   = DEFAULT_MAX_MEL_LENGTH,
                 max_output_length  = DEFAULT_MAX_TEXT_LENGTH,
                 
                 use_ctc_decoder    = None,
                 
                 ** kwargs
                ):
        self._init_text(lang = lang, ** kwargs)
        self._init_audio(** kwargs)
        
        self.use_fixed_length_input = use_fixed_length_input
        self.max_input_length   = max_input_length
        self.max_output_length  = max_output_length
        
        super().__init__(** kwargs)
        
        if hasattr(self.model, 'set_tokens'): self.model.set_tokens(** self.model_tokens)

    def build(self, architecture = None, *, model = None, stt_model = None, ** kwargs):
        if stt_model is not None: model = stt_model
        elif model is None:
            assert architecture, 'You must specify the architecture name'
            
            model = {
                'architecture'  : architecture,
                'input_shape'   : self.mel_input_shape,
                'vocab_size'    : self.vocab_size,
                'pad_value'     : self.pad_mel_value,
                'sos_token'     : self.sos_token_idx,
                'eos_token'     : self.eos_token_idx,
                'pad_token'     : self.blank_token_idx,
                ** kwargs
            }

        super().build(model = model)
    
    @property
    def use_ctc_decoder(self):
        return not self.is_encoder_decoder
    
    @property
    def mel_input_shape(self):
        mel_length = self.max_input_length if self.use_fixed_length_input else None
        return (mel_length, ) + self.audio_signature.shape[2:]
    
    @property
    def input_signature(self):
        inp_sign = TensorSpec(shape = (None, ) + self.mel_input_shape, dtype = 'float32')
        
        return inp_sign if not self.is_encoder_decoder else (inp_sign, self.text_signature)
    
    @property
    def infer_signature(self):
        return TensorSpec(shape = (None, ) + self.mel_input_shape, dtype = 'float32')

    @property
    def training_hparams(self):
        return {
            ** super().training_hparams,
            ** self.training_hparams_audio
        }
    
    def __str__(self):
        des = super().__str__()
        des += self._str_text()
        des += self._str_audio()
        des += "- Use CTC decoder : {}\n".format(self.use_ctc_decoder)
        return des
    
    def _infer_segments(self, mel, time_window = 30, segment_processing = None, ** kwargs):
        window_samples = self._get_sample_index(time_window)
        
        segments, total_text = [], ''
        for i, start in enumerate(range(0, len(mel), window_samples)):
            segment = self.pad_or_trim(mel[start : start + window_samples])
            
            tokens = None
            if self.is_encoder_decoder:
                tokens = self.get_inference_tokens(prev_text = total_text, ** kwargs)
            
            pred    = self.compiled_infer(
                ops.expand_dims(segment, axis = 0), tokens = tokens, ** kwargs
            )
            tokens  = getattr(pred, 'tokens', pred)
            
            text    = self.decode_output(pred) if len(pred) > 0 else ''
            while isinstance(text, list): text = text[0]
            text    = text.strip()
            
            infos = {
                "start" : start,
                "end"   : end,
                "time"  : end - start,
                "text"  : text,
                "tokens"    : tokens,
                "score"     : ops.convert_to_numpy(pred.score[0]) if hasattr(result, 'score') else 0
            }
            if lang: infos['lang'] = lang
            
            if segment_processing is not None:
                segment_processing(infos, segment = segment)
            
            segments.append(infos)
            if i > 0: total_text += ' '
            total_text += text
        
        return segments

    @timer(name = 'inference')
    def infer(self,
              audio,
              *,
              
              verbose   = False,
              
              callbacks = None,
              predicted = None,
              overwrite = False,
              return_output = True,
              
              condition_on_previous_text    = True,
              
              ** kwargs
             ):
        if predicted and not overwrite and isinstance(audio, str) and audio in predicted:
            if callbacks: apply_callbacks(callbacks, predicted[audio], {}, save = False)
            return predicted[audio]
        
        if isinstance(audio, str) and verbose:
            logger.info('Processing file {}...'.format(audio))

        infos = audio.copy() if isinstance(audio, dict) else {}
        
        with Timer('loading audio'):
            mel = self.get_input(audio, pad_or_trim = False, ** kwargs)
        
        segments = self._infer_segments(
            mel, verbose = verbose, condition_on_previous_text = condition_on_previous_text, ** kwargs
        )

        infos.update({
            'text'      : ' '.join([seg['text'] for seg in segments]),
            'segments'  : segments
        })
        if 'lang' not in infos and segments and 'lang' in segments[0]:
            infos['lang'] = segments[0]['lang']
        
        if not isinstance(audio, dict):
            if isinstance(audio, str): infos['filename'] = audio
            else:                      infos['audio'] = audio
        
        if callbacks:
            entry = apply_callbacks(
                callbacks, {k : v for k, v in infos.items() if k != 'audio'}, infos, save = True
            )
        
        return infos if return_output else {k : v for k, v in infos.items() if k != 'audio'}
    
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
    
    def decode_output(self, output, ** kwargs):
        if self.use_ctc_decoder:
            return self.ctc_decode_text(output, ** kwargs)
        return self.decode_text(output, ** kwargs)
    
    def pad_or_trim(self, audio):
        if ops.shape(audio)[0] > self.max_input_length:
            audio = audio[: self.max_input_length]
        elif self.use_fixed_length_input and ops.shape(audio)[0] != self.max_input_length:
            audio = ops.pad(
                audio, [(0, self.max_input_length - ops.shape(audio)[0]), (0, 0)],
                constant_values = self.pad_mel_value
            )
        
        return audio
    
    def prepare_input(self, data, pad_or_trim = True, ** kwargs):
        audio = self.get_audio(data, ** kwargs)
        
        return self.pad_or_trim(audio) if pad_or_trim else audio

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
    
    def get_inference_callbacks(self,
                                *,

                                save    = True,
                                display = None,
                             
                                directory   = None,
                                raw_audio_dir   = None,
                                filename    = 'audio_{}.mp3',
                                 
                                post_processing     = None,
                                 
                                save_in_parallel    = False,

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
                - display   : verbosity level (cumulative, i.e., level 2 includes level 1)
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
        if save is None:    save = not display
        if display is None: display = not save
        
        if directory is None: directory = self.pred_dir
        map_file    = os.path.join(directory, 'map.json')
        
        predicted   = {}
        callbacks   = []
        if save:
            predicted   = load_json(map_file, {})
            
            if raw_audio_dir is None: raw_audio_dir = os.path.join(directory, 'audios')
            callbacks.append(AudioSaver(
                key = 'filename',
                name    = 'saving raw',
                data_key    = 'audio',
                file_format = os.path.join(raw_audio_dir, filename),
                index_key   = 'segment_index',
                save_in_parallel  = save_in_parallel
            ))
            
            callbacks.append(JSonSaver(
                data    = predicted,
                filename    = map_file,
                primary_key = 'filename',
                save_in_parallel = save_in_parallel
            ))
        
        if display:
            callbacks.append(AudioDisplayer(show_text = True))
        
        if post_processing is not None:
            if not isinstance(post_processing, list): post_processing = [post_processing]
            for fn in post_processing:
                if callable(fn):
                    callbacks.append(FunctionCallback(fn))
                elif hasattr(fn, 'put'):
                    callbacks.append(QueueCallback(fn))
        
        return predicted, callbacks

    @timer
    def predict(self, inputs, ** kwargs):
        if (isinstance(inputs, (str, dict))) or (ops.is_array(inputs) and len(inputs.shape) == 1):
            inputs = [inputs]
        
        return super().predict(inputs, ** kwargs)

    def stream(self, stream, ** kwargs):
        # used to compile the mel-spectrogram to avoid warmup during effective stream
        for length in (self.rate // 2, self.rate):
            self.get_input({'audio' : np.zeros((length, ), dtype = 'float32'), 'rate' : self.rate})
        
        return super().stream(stream, ** kwargs)
    
    def get_config(self):
        return {
            ** super().get_config(),
            ** self.get_config_audio(),
            ** self.get_config_text(),

            'use_fixed_length_input'    : self.use_fixed_length_input,
            'max_input_length'  : self.max_input_length,
            'max_output_length' : self.max_output_length
        }
