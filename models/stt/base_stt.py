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

from loggers import timer, time_logger
from utils.keras_utils import TensorSpec, ops
from models.interfaces.base_text_model import BaseTextModel
from models.interfaces.base_audio_model import BaseAudioModel
from utils import dump_json, load_json, normalize_filename, pad_batch, get_filename, should_predict
from utils.audio import write_audio, load_audio, display_audio, AudioAnnotation, AudioSearch, SearchResult
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
        
        if hasattr(self.stt_model, 'set_tokens'): self.stt_model.set_tokens(** self.model_tokens)

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
    def is_encoder_decoder(self):
        if hasattr(self, 'stt_model'):
            return getattr(self.stt_model, 'decoder', None) is not None
        raise NotImplementedError(
            'You must define `is_encoder_decoder` because it is required before building the model !'
        )
    
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
    
    @timer(name = 'prediction', log_if_root = False)
    def call(self, inputs, training = False, ** kwargs):
        """
            Arguments : 
                - inputs :
                    if `self.is_encoder_decoder`:
                        - audio / mel   : [B, seq_in_len, n_channels]
                        - text          : [B, seq_out_len]
                    else :
                        - audio / mel   : [B, seq_len, n_channels]
            Return :
                - outputs   : the score for each token at each timestep
                    if `self.is_encoder_decoder` :
                        - shape = [B, sub_seq_len, vocab_size]
                        Note that `sub_seq_len` may differ from `seq_len` (due to subsampling)
                    else :
                        - shape = [B, seq_out_len, vocab_size]
        """
        if self.stt_model.__class__.__name__ == 'Functional': kwargs = {}
        return self.stt_model(inputs, training = training, ** kwargs)
    
    @timer(name = 'inference', log_if_root = False)
    def infer(self, inputs, training = False, decode = False, prev_tokens = None, ** kwargs):
        if hasattr(self.stt_model, 'infer'):
            output = self.stt_model.infer(inputs, training = training, ** kwargs)
        else:
            output = self(inputs, training = training, ** kwargs)
        
        return self.decode_output(output) if decode else output
    
    def encode_audio(self, inputs, training = False, ** kwargs):
        if self.is_encoder_decoder:
            return self.stt_model.encoder(inputs, training = training, ** kwargs)
        return self(inputs, training = training, ** kwargs)
    
    def decode_audio(self, encoder_output, training = False, return_state = False, ** kwargs):
        if len(ops.shape(encoder_output)) == 2:
            encoder_output = ops.expand_dims(encoder_output, axis = 0)
        
        if self.use_ctc_decoder:
            return encoder_output if not return_state else (encoder_output, None)

        return self.stt_model.infer(
            encoder_output = encoder_output, training = training, return_state = return_state,
            ** kwargs
        )
    
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
        
    def predict_with_target(self, batch, epoch = None, step = None, prefix = None, 
                            directory = None, n_pred = 1):
        inputs, output = batch
        inputs = [inp[:n_pred] for inp in inputs]
        output = [out[:n_pred] for out in output]
        
        infer   = None
        pred    = self.decode_output(self(inputs, training = False))
        if self.is_encoder_decoder:
            infer = self.infer(
                inputs[:-1], early_stopping = False, max_length = ops.shape(output)[1], decode = True
            )
        
        target = self.decode_output(output)
        
        des = ""
        for i in range(len(target)):
            des = "  Target     : {}\n  Prediction : {}{}\n".format(
                target[i], pred[i],
                "" if infer is None else "\n  Inference  : {}".format(infer[i])
            )
        logger.info(des)

    def _add_segment(self, all_segments, segment, start, end, tokens, result, ** kwargs):
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
        if text: all_segments.append(infos)

        return post_process(segment, infos, ** kwargs)

    def predict_segment(self, mel, start, end, time_window = 30, ** kwargs):
        window = self._get_sample_index(time_window)
        
        segments = []
        for i, start in enumerate(range(0, len(mel), window)):
            pred = self.infer(
                ops.expand_dims(mel[start : start + window], axis = 0), decode = False, ** kwargs
            )

            self._add_segment(
                segments,
                segment = mel[start : start + window],
                start   = time_window * i,
                end     = time_window * (i + 1),
                tokens  = pred if not hasattr(pred, 'tokens') else pred.tokens,
                result  = pred,
                ** kwargs
            )
        
        return segments
    
    @timer
    def predict(self,
                filenames,
                alignments  = None,
                
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
        #  Initialization  #
        ####################
        
        with time_logger.timer('initialization'):
            if not isinstance(filenames, (list, tuple, pd.DataFrame)): filenames = [filenames]

            if directory is None: directory = self.pred_dir
            raw_audio_dir   = os.path.join(directory, 'audios')
            map_file    = os.path.join(directory, 'map.json')

            predicted = {}
            if save:
                os.makedirs(directory, exist_ok = True)

                predicted = load_json(map_file, default = {})

        with time_logger.timer('pre_processing'):
            results = [None] * len(filenames)
            duplicatas  = {}
            requested   = [(
                get_filename(audio, keys = ('filename', )) if not hasattr(audio, 'filename') else audio.filename,
                audio
            ) for audio in filenames]

            inputs = []
            for i, (file, audio) in enumerate(requested):
                if not should_predict(predicted, file, overwrite = overwrite, timestamp = timestamp):
                    if verbose: logger.info('Audio {} already processed'.format(file))
                    results[i] = (file, predicted[file])
                    continue

                if isinstance(file, str):
                    duplicatas.setdefault(file, []).append(i)

                    if len(duplicatas[file]) > 1:
                        continue

                inputs.append((i, file, audio))

        
        for idx, file, data in inputs:
            if isinstance(file, str) and verbose:
                logger.info('Processing file {}...'.format(file))

            with time_logger.timer('loading audio'):
                mel = self.get_input(data, pad_or_trim = False)

            # Get associated alignment (if provided)
            alignment = alignments[idx] if alignments is not None else None
            
            if isinstance(data, AudioAnnotation):
                alignment = data._alignment
            elif isinstance(data, dict):
                alignment = data.get('alignment', alignment)

            if alignment is None: alignment = [{'id' : -1, 'start' : 0., 'end' : None}]
            
            all_segments = []
            for align in alignment:
                segments = self.predict_segment(
                    mel,
                    verbose = verbose,
                    post_processing = post_processing,
                    ** {** kwargs, ** align}
                )
                if isinstance(segments, dict): segments = [segments]
                for segment in segments:
                    all_segments.append({** align, ** segment})
                    all_segments[-1]['start']   += align['start']
                    all_segments[-1]['end']     += align['start']
            
            infos = {
                'filename'  : file,
                'text'  : ' '.join([seg['text'] for seg in all_segments]),
                'alignment' : all_segments
            }
            
            if file and file in duplicatas:
                for idx in duplicatas[file]:
                    results[idx] = (data, infos)
            else:
                results[i] = (data, infos)
            
            if save:
                if not file:
                    with time_logger.timer('saving audios'):
                        os.makedirs(raw_audio_dir, exist_ok = True)
                        file = os.path.join(raw_audio_dir, raw_audio_filename)
                        if '{}' in file:
                            file = file.format(len(glob.glob(file.replace('{}', '*'))))

                        write_audio(filename = file, audio = data, rate = self.audio_rate)

                infos['filename']   = file
                predicted[file]     = infos
                
                with time_logger.timer('saving json'):
                    dump_json(map_file, predicted, indent = 4)
        
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

def post_process(segment, infos, post_processing = None, verbose = True, ** kwargs):
    if verbose == 2:
        logger.info('Add segment from {} to {} with text {}'.format(
            infos['start'], infos['end'], infos['text']
        ))

    if post_processing is not None:
        try:
            post_processing(segment, infos, ** kwargs)
        except Exception as e:
            logger.error('Exception in `post_processing` : {}'.format(e))

    return infos
