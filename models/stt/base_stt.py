
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
import time
import logging
import numpy as np
import tensorflow as tf

from tqdm import tqdm

from loggers import timer
from utils.thread_utils import Pipeline
from models.interfaces.base_text_model import BaseTextModel
from models.interfaces.base_audio_model import BaseAudioModel
from utils import dump_json, load_json, normalize_filename, pad_batch
from utils.audio import write_audio, load_audio, AudioAnnotation, AudioSearch, SearchResult
from utils.text import get_encoder, get_symbols, accent_replacement_matrix, decode

logger      = logging.getLogger(__name__)
time_logger = logging.getLogger('timer')

_silent_char    = list(" '\"")
_deep_speech_en_symbols = list(" abcdefghijklmnopqrstuvwxyz'") + ['']

MIN_AUDIO_TIME  = 0.1

DEFAULT_MAX_MEL_LENGTH  = 1024
DEFAULT_MAX_TEXT_LENGTH = min(256, DEFAULT_MAX_MEL_LENGTH // 2)
        
class BaseSTT(BaseTextModel, BaseAudioModel):
    def __init__(self,
                 lang,
                 use_ctc_decoder,
                 
                 text_encoder   = None,
                 
                 use_fixed_length_input = False,
                 max_input_length   = DEFAULT_MAX_MEL_LENGTH,
                 max_output_length  = DEFAULT_MAX_TEXT_LENGTH,
                 
                 ** kwargs
                ):
        text_encoder = get_encoder(
            text_encoder = text_encoder,
            lang    = lang,
            vocab   = get_symbols(
                lang, maj = False, arpabet = False, punctuation = 2
            ),
            use_sos_and_eos = not use_ctc_decoder
        )

        self._init_text(lang = lang, text_encoder = text_encoder, ** kwargs)
        
        self._init_audio(** kwargs)
        
        self.use_ctc_decoder    = use_ctc_decoder
        
        self.use_fixed_length_input = use_fixed_length_input
        self.max_input_length   = max_input_length
        self.max_output_length  = max_output_length
        
        super().__init__(** kwargs)
    
    def _init_folders(self):
        super()._init_folders()
        os.makedirs(self.search_dir, exist_ok = True)

    def _build_model(self, architecture_name, ** kwargs):
        super(BaseSTT, self)._build_model(
            stt_model = {
                'architecture_name' : architecture_name,
                'input_shape'   : self.mel_input_shape,
                'vocab_size'    : self.vocab_size,
                ** kwargs
            }
        )
            
    @property
    def search_dir(self):
        return os.path.join(self.folder, 'search')
    
    @property
    def pred_map_file(self):
        return os.path.join(self.pred_dir, 'map.json')
    
    @property
    def mel_input_shape(self):
        mel_length = self.max_input_length if self.use_fixed_length_input else None
        return (mel_length, ) + self.audio_signature.shape[2:]
    
    @property
    def decoder_method(self):
        if not self.use_ctc_decoder: return 'greedy'
        return 'beam_search'
    
    @property
    def input_signature(self):
        audio_sign  = (
            tf.TensorSpec(shape = (None, ) + self.mel_input_shape, dtype = tf.float32),
            tf.TensorSpec(shape = (None,), dtype = tf.int32)
        )
        
        if self.use_ctc_decoder:
            return audio_sign
        
        return audio_sign + self.text_signature
        
    @property
    def output_signature(self):
        return self.text_signature
        
    @property
    def training_hparams(self):
        return super().training_hparams(
            ** self.training_hparams_audio,
            max_input_length    = None,
            max_output_length   = None,
        )
    
    @property
    def sep_token(self):
        return '-'
    
    @property
    def is_encoder_decoder(self):
        return True if hasattr(self.stt_model, 'decoder') else False
    
    def __str__(self):
        des = super().__str__()
        des += self._str_text()
        des += self._str_audio()
        des += "Use CTC decoder : {}\n".format(self.use_ctc_decoder)
        return des
    
    @timer(name = 'prediction', log_if_root = False)
    def call(self, inputs, training = False):
        """
            Arguments : 
                - inputs :
                    if self.use_ctc_decoder : (mel, mel_length)
                        - mel           : [B, seq_len, n_mel_channels]
                        - mel_length    : [B, 1]
                    else : (mel, mel_length, text, text_length)
                        - mel           : [B, seq_len, n_mel_channels]
                        - mel_length    : [B, 1]
                        - text          : [B, out_seq_len]
                        - text_length   : [B, 1]
            Return :
                if self.use_ctc_decoder : outputs
                else : (outputs, attn_weights)
                
                    - outputs : [B, out_seq_len, vocab_size] : score for each token at each timestep
        """
        return self.stt_model(inputs, training = training)
    
    @timer(name = 'inference', log_if_root = False)
    def infer(self, inputs, training = False, decode = False, ** kwargs):
        if self.use_ctc_decoder:
            output = self(inputs, training = training, ** kwargs)
        else:
            output = self.stt_model.infer(inputs, training = training, ** kwargs)
        
        return self.decode_output(output) if decode else output
    
    def encode(self, inputs, training = False, ** kwargs):
        if self.use_ctc_decoder:
            return self(inputs, training = training, ** kwargs)
        return self.stt_model.encoder(inputs, training = training, ** kwargs)
    
    def decode(self, encoder_output, training = False, return_state = False, ** kwargs):
        if len(tf.shape(encoder_output)) == 2:
            encoder_output = tf.expand_dims(encoder_output, axis = 0)
        
        if self.use_ctc_decoder:
            return encoder_output if not return_state else (encoder_output, None)

        return self.stt_model.infer(
            encoder_output = encoder_output, training = training, return_state = return_state,
            ** kwargs
        )
    
    def compile(self, loss = None, metrics = None, ** kwargs):
        if loss is None:
            loss = 'CTCLoss' if self.use_ctc_decoder else 'TextLoss'
        if metrics is None:
            if self.use_ctc_decoder:
                metrics = [{'metric' : 'TextMetric', 'config': {'pad_value' : self.blank_token_idx}}]
            else:
                metrics = ['TextAccuracy']
            
        super().compile(loss = loss, metrics = metrics, ** kwargs)
    
    def decode_output(self, output, * args, ** kwargs):
        if len(output.shape) in (1, 2):
            return self.decode_text(output)
        elif len(output.shape) == 3:
            pred = decode(
                output, method = self.decoder_method, blank_idx = self.blank_token_idx, ** kwargs
            )
            return [self.decode_output(p) for p in pred]
        else:
            raise ValueError("Invalid shape : {}".format(output.shape))
    
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
    
    def get_input(self, data):
        return self.get_audio(data)
    
    def encode_data(self, data):
        encoded_text = self.tf_encode_text(data)
        
        mel = self.get_input(data)
        
        return mel, len(mel), encoded_text, len(encoded_text)
    
    def filter_data(self, mel, mel_length, text, text_length):
        return tf.logical_and(
            mel_length <= self.max_input_length, 
            text_length <= self.max_output_length
        )
        
    def augment_data(self, mel, mel_length, text, text_length):
        mel = self.augment_audio(mel, max_length = self.max_input_length)
        
        return mel, len(mel), text, text_length
        
    def preprocess_data(self, mel, mel_length, text, text_length):
        if self.use_ctc_decoder:
            return (mel, mel_length), (text, text_length)
        
        text_in, text_out = text, text
        text_in_len, text_out_len   = text_length, text_length
        if self.text_encoder.use_sos_and_eos:
            text_in, text_in_len    = text_in[:, :-1], text_in_len - 1
            text_out, text_out_len  = text_out[:, 1:], text_out_len - 1
        
        return (mel, mel_length, text_in, text_in_len), (text_out, text_out_len)
        
    def get_dataset_config(self, ** kwargs):
        kwargs.update({
            'batch_before_map'  : True,
            'padded_batch'  : True,
            'pad_kwargs'    : {
                'padding_values' : (self.pad_mel_value, 0, self.blank_token_idx, 0)
            }
            
        })
        if self.use_fixed_length_input:
            kwargs['pad_kwargs']['padded_shapes'] = (
                self.mel_input_shape, (), (None,), ()
            )
        
        return super().get_dataset_config(** kwargs)
        
    def train_step(self, batch):
        inputs, target = batch
        
        with tf.GradientTape() as tape:
            pred = self(inputs, training = True)
            if isinstance(pred, tuple): pred, _ = pred
            
            loss = self.stt_model_loss(target, pred)
        
        variables = self.stt_model.trainable_variables

        grads = tape.gradient(loss, variables)
        self.stt_model_optimizer.apply_gradients(zip(grads, variables))
        
        return self.update_metrics(target, pred)
        
    def eval_step(self, batch):
        inputs, target = batch

        pred = self(inputs, training = False)
        if isinstance(pred, tuple): pred, _ = pred
        
        return self.update_metrics(target, pred)
    
    def predict_with_target(self, batch, epoch = None, step = None, prefix = None, 
                            directory = None, n_pred = 1):
        inputs, output = batch
        inputs = [inp[:n_pred] for inp in inputs]
        text, max_length = [out[:n_pred] for out in output]
        
        if self.use_ctc_decoder:
            pred = self(inputs, training = False)
            pred = self.decode_output(pred)
            infer = None
        else:
            pred = self(inputs, training = False)
            infer = self.infer(
                inputs[:-2] if not self.use_ctc_decoder else inputs,
                max_length      = max_length,
                early_stopping  = False,
                decode  = True
            )
        
            pred    = self.decode_output(pred)
            infer   = self.decode_output(infer)
                
        target = self.decode_output(text)
        
        des = ""
        for i in range(len(target)):
            des = "  Target     : {}\n  Prediction : {}{}\n".format(
                target[i], pred[i],
                "" if infer is None else "\n  Inference  : {}".format(infer[i])
            )
        logger.info(des)
    
    def get_pipeline(self,
                     cumulative = False,
                     max_time   = -1,
                     time_window    = 5.,
                     max_frames = -1,
                     
                     save   = True,
                     directory  = None,
                     
                     processing_fn  = None,
                     
                     playback   = False,
                     
                     ** kwargs
                    ):
        def _micro_audio_stream(stream):
            p = pyaudio.PyAudio()
            inp_stream = p.open(
                format = pyaudio.paFloat32, channels = 1, rate = self.audio_rate, input = True
            )

            print('Start recording...')
            for i in range(max_time // time_window + 1):
                yield np.frombuffer(inp_stream.read(frames_per_buffer), dtype = np.float32)
            print('\nStop recording !')
            inp_stream.close()

        def _file_audio_stream(stream):
            audio = load_audio(stream, self.audio_rate)
            
            end = min(int(max_time * self.audio_rate), len(audio)) if max_time > 0 else len(audio)
            for i, start in enumerate(range(0, end, frames_per_buffer)):
                yield audio[start : start + frames_per_buffer]

        def frames_generator(stream, ** kw):
            def frame_iterator():
                generator_fn = _file_audio_stream if isinstance(stream, str) else _micro_audio_stream
                cumulated_audio = None
                for i, frame in enumerate(generator_fn(stream)):
                    if cumulative:
                        cumulated_audio = frame if cumulated_audio is None else np.concatenate(
                            [cumulated_audio, frame], axis = 0
                        )
                    
                    if not isinstance(frame, dict):
                        frame = {
                            'audio' : cumulated_audio if cumulative else frame,
                            'frame_index'   : i,
                            'start' : 0 if cumulative else time_window * i,
                            'end'   : time_window * (i + 1)
                        }
                    yield frame
                yield None
            
            generator_fn = _file_audio_stream if isinstance(stream, str) else _micro_audio_stream
            return (stream, frame_iterator())

        @timer
        def audio_to_mel(infos, ** kw):
            if infos is None: return None
            inputs = tf.expand_dims(self.get_input(infos.pop('audio')), axis = 0)
            return (infos, inputs)

        @timer
        def encode(data, ** kw):
            if data is None: return None
            
            infos, inputs = data if not isinstance(data, list) else list(zip(* data))
            if not isinstance(inputs, (list, tuple)) and max_frames > 0 and tf.shape(inputs)[1] > max_frames:
                logger.info('Too many frames ({}), truncating to {}'.format(
                    tf.shape(inputs)[1], max_frames
                ))
                inputs = inputs[:, - max_frames :]

            encoder_output = self.encode(inputs, training = False)
            if isinstance(encoder_output, (list, tuple)): encoder_output = encoder_output[0]

            if isinstance(data, list):
                return [(info, out) for info, out in zip(infos, encoder_output)]
            return infos, encoder_output[0]

        @timer
        def decode(data, last_tokens = None, state = None, ** kw):
            if data is None:
                return None, (last_tokens, state)

            infos, encoder_output = data
            
            outputs = self.decode(
                tokens = last_tokens,
                encoder_output = encoder_output,

                initial_state   = state if not cumulative else None,
                return_state = True if not cumulative else False,
                ** kwargs
            )
            state = None
            if not cumulative:
                if hasattr(outputs, 'state'):
                    state = outputs.state
                elif isinstance(outputs, tuple):
                    outputs, state = outputs
            
            tokens = outputs.tokens if hasattr(outputs, 'tokens') else None
            
            infos['text'] = self.decode_output(outputs)
            return infos, (tokens, state)

        @timer
        def show_output(infos, ** kw):
            if infos is None:
                print()
                return
            
            print(infos['text'])

        if processing_fn is None: processing_fn = audio_to_mel
        
        frames_per_buffer = int(time_window * self.audio_rate)
        
        playback_fn = None
        if playback:
            p = pyaudio.PyAudio()
            out_stream = p.open(
                format = pyaudio.paFloat32, channels = 1, rate = self.audio_rate, output = True,
                frames_per_buffer = frames_per_buffer
            )
            
            playback_fn = {
                'name'  : 'playback',
                'consumer' : lambda infos, ** kw: out_stream.write(infos['audio'].tobytes()) if infos is not None else None,
                'allow_multithread' : False,
                'stop_listeners'    : out_stream.close
            }

        if directory is None: directory = self.pred_dir
        
        pipeline = Pipeline(** {
            'name'      : 'transcription_pipeline',
            'filename'  : os.path.join(directory, 'map.json') if save else None,
            
            'tasks' : [
                {
                    'consumer' : frames_generator, 'splitter' : True, 'consumers' : playback_fn
                },
                processing_fn,
                encode,
                {
                    'consumer'  : decode,
                    'stateful'  : True,
                    'consumers' : show_output
                },
                {'consumer' : 'grouper', 'nested_group' : False}
            ],
            ** kwargs
        })
        pipeline.start()
        return pipeline
    
    @timer
    def predict(self,
                filenames,
                alignments  = None,
                time_window = 30,
                time_step   = 27.5, 
                batch_size  = 8,
                
                max_err = 3,
                
                save    = True,
                directory   = None,
                overwrite   = False,
                
                verbose = 1,
                tqdm    = tqdm,
                ** kwargs
               ):
        @timer(name = 'post processing')
        def combine_text(alignments, max_err):
            if len(alignments) == 0: return ''
            if max_err < 1: max_err = int(max_err * 100.)
            
            text = alignments[0]['text']
            for i in range(1, len(alignments)):
                last_text, new_text = alignments[i-1]['text'], alignments[i]['text']
                
                overlap = alignments[i]['start'] - alignments[i-1]['end']
                if overlap <= 0:
                    text += ' ' + new_text
                    continue
                
                prop    = overlap / alignments[i]['time']
                
                min_length = int(prop * len(new_text) / 2)
                max_length = int(prop * len(new_text) * 1.2)
                
                best_idx, best_dist, stop = 0, 1, False
                for j in reversed(range(min_length+1, max_length)):
                    end, new_end = last_text[-j :], new_text[: j]
                    
                    if len(end) != len(new_end) or end[0] != new_end[0]: continue
                    
                    dist = self.distance(end, new_end)
                    max_dist = max_err / j
                    
                    if dist < max_dist: stop = True
                    if dist < max_dist * 2 and dist < best_dist:
                        best_idx, best_dist = j, dist
                    
                    if stop and dist > max_dist: break

                mid     = best_idx // 2 if best_idx % 2 != 0 else best_idx // 2 + 1
                
                text    = text[: -mid] + ' ' + text[mid :]

            return text
            
        # Normalize variables
        filenames = normalize_filename(filenames, invalid_mode = 'keep')
        
        if not isinstance(filenames, (list, tuple)):
            if alignments is not None: alignments = [alignments]
            filenames = [filenames]
        
        # Define directories
        if directory is None: directory = self.pred_dir
        map_file    = os.path.join(directory, 'map.json')
        audio_dir   = os.path.join(directory, 'audios')
        if save: os.makedirs(audio_dir, exist_ok = True)
        
        all_outputs = {}
        if os.path.exists(map_file):
            all_outputs = load_json(map_file)
        
        outputs = []
        for i, filename in enumerate(filenames):
            # Get associated alignment
            alignment = alignments[i] if alignments is not None else None
            
            if isinstance(filename, AudioAnnotation):
                filename, alignment = filename.filename, filanem._alignment
            elif isinstance(filename, dict):
                filename, alignment = filename['filename'], filename.get('alignment', alignment)
            # Load (or save) audio
            if isinstance(filename, str):
                if filename in all_outputs and not overwrite:
                    outputs.append(all_outputs[filename])
                    continue
                
                time_logger.start_timer('processing')
                audio = self.get_audio_input(filename)[:, 0]
                time_logger.stop_timer('processing')
            else:
                audio = filename
                if save:
                    filename = 'audio_{}.wav'.format(len(os.listdir(audio_dir)))
                    write_audio(audio = audio, filename = filename, rate = self.audio_rate)
            
            logger.info("Processing file {}...".format(filename))
            
            audio_time = int(len(audio) / self.audio_rate)
            if alignment is None:
                alignment = [{
                    'start' : t,
                    'end'   : min(t + time_window, audio_time),
                    'text'  : ''
                } for t in np.arange(0, audio_time, time_step)]
            
            time_logger.start_timer('processing')
            # Slice audio by step with part of length `window` (and remove part < 0.1 sec)
            inputs = [audio[
                int(info['start'] * self.audio_rate) : int(info['end'] * self.audio_rate)
            ] for info in alignment]
            inputs = [
                self.get_audio(a) for a in inputs if len(a) > self.audio_rate * MIN_AUDIO_TIME
            ]
            time_logger.stop_timer('processing')
            
            # Get text result 
            text_outputs = []
            for b in tqdm(range(0, len(inputs), batch_size)):
                mels    = inputs[b : b + batch_size]
                length  = tf.cast([len(mel) for mel in mels], tf.int32)
                
                batch   = [pad_batch(mels, pad_value = self.pad_mel_value), length]
                pred    = self.infer(batch, decode = True)
                if isinstance(pred, tuple): pred, _ = pred
                
                text_outputs += pred
            
            for info, text in zip(alignment, text_outputs):
                info.setdefault('id', -1)
                info.update({
                    'text'  : text.strip(),
                    'time'  : info['end'] - info['start']
                })
            
            audio_infos = {
                'filename'  : filename,
                'time'      : len(audio) / self.audio_rate,
                'text'      : combine_text(alignment, max_err),
                'alignment' : alignment
            }
            outputs.append(audio_infos)
            
            if isinstance(filename, str): all_outputs[filename] = audio_infos
        
        # Save prediction to map file
        if save:
            dump_json(map_file, all_outputs, indent = 4)
        
        # Return desired outputs as a list of dict
        return outputs
    
    @timer
    def stream(self, max_time = 30, filename = None):
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

        pred = self.infer(audio, decode = True)
        text = pred[0] if self.use_ctc_decoder else pred[0][0]
        
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
        ) for infos in pred])
    
    def get_config(self, * args, ** kwargs):
        config = super().get_config(* args, ** kwargs)    
        config.update({
            ** self.get_config_audio(),
            ** self.get_config_text(),
            'use_ctc_decoder'   : self.use_ctc_decoder,
            
            'use_fixed_length_input'    : self.use_fixed_length_input,
            'max_input_length'  : self.max_input_length,
            'max_output_length' : self.max_output_length
        })
        
        return config
