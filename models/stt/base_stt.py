import os
import time
import logging
import datetime
import itertools
import numpy as np
import tensorflow as tf

from tqdm import tqdm

from loggers import timer
from models.base_model import BaseModel
from custom_architectures import get_architecture
from models.weights_converter import partial_transfer_learning
from utils import dump_json, load_json, normalize_filename, pad_batch
from utils.audio import load_mel, load_audio, write_audio
from utils.audio import MelSTFT, DeepSpeechSTFT, JasperSTFT, AudioAnnotation, AudioSearch, SearchResult
from utils.text import get_encoder, get_symbols, TextEncoder, accent_replacement_matrix, decode

time_logger = logging.getLogger('timer')

_silent_char    = list(" '\"")
_deep_speech_en_symbols = list(" abcdefghijklmnopqrstuvwxyz'") + ['']

MIN_AUDIO_TIME  = 0.1

DEFAULT_MAX_MEL_LENGTH  = 1024
DEFAULT_MAX_TEXT_LENGTH = min(256, DEFAULT_MAX_MEL_LENGTH // 2)

DEFAULT_MEL_FRAMES      = 64

DEFAULT_MEL_FN_CONFIG  = {
    'filter_length'    : 1024,
    'hop_length'       : 256, 
    'win_length'       : 1024,
    'n_mel_channels'   : 80,
    'sampling_rate'    : 22050, 
    'mel_fmin'         : 0.0,
    'mel_fmax'         : 8000.0,
    'normalize_mode'   : None
}
        
class BaseSTT(BaseModel):
    def __init__(self,
                 lang,
                 mel_as_image,
                 use_ctc_decoder,
                 
                 text_encoder       = None,
                 
                 use_fixed_length_input = False,
                 max_input_length   = DEFAULT_MAX_MEL_LENGTH,
                 max_output_length  = DEFAULT_MAX_TEXT_LENGTH,
                 
                 pad_mel_value      = 0.,
                 mel_fn_type        = 'TacotronSTFT',
                 mel_fn_config      = DEFAULT_MEL_FN_CONFIG,
                 
                 **kwargs
                ):        
        self.lang   = lang
        self.pad_mel_value  = pad_mel_value
        self.mel_as_image   = mel_as_image
        self.use_ctc_decoder    = use_ctc_decoder
        
        self.use_fixed_length_input = use_fixed_length_input
        self.max_input_length   = max_input_length
        self.max_output_length  = max_output_length
        
        
        self.text_encoder = get_encoder(
            text_encoder = text_encoder,
            lang    = lang,
            vocab   = get_symbols(
                lang, maj = False, arpabet = False, punctuation = 2
            ),
            use_sos_and_eos = not self.use_ctc_decoder
        )
            
        # Initialization of mel fn
        if isinstance(mel_fn_type, MelSTFT):
            self.mel_fn = mel_fn_type
        else:
            self.mel_fn    = MelSTFT.create(mel_fn_type, ** mel_fn_config)
        
        super().__init__(** kwargs)
        
        if not os.path.exists(self.text_encoder_file):
            self.text_encoder.save_to_file(self.text_encoder_file)
        if not os.path.exists(self.mel_fn_file):
            self.mel_fn.save_to_file(self.mel_fn_file)
    
        if hasattr(self.stt_model, '_build'): self.stt_model._build()
    
    def _init_folders(self):
        super()._init_folders()
        os.makedirs(self.search_dir, exist_ok = True)

    def init_train_config(self,
                          max_input_length  = None,
                          max_output_length = None,
                          
                          trim_audio   = False,
                          reduce_noise = False,
                          trim_threshold   = 0.1,
                          max_silence  = 0.15,
                          trim_method  = 'window',
                          trim_mode    = 'start_end',
                          
                          trim_mel     = False,
                          trim_factor  = 0.6,
                          trim_mel_method  = 'max_start_end',
                          ** kwargs
                          ):
        if max_input_length: self.max_input_length   = max_input_length
        if max_output_length: self.max_output_length  = max_output_length
                
        self.trim_audio     = trim_audio
        self.trim_kwargs    = {
            'trim_silence'  : trim_audio,
            'reduce_noise'  : reduce_noise,
            'method'    : trim_method,
            'mode'      : trim_mode,
            'threshold' : tf.cast(trim_threshold, tf.float32),
            'max_silence'   : tf.cast(max_silence, tf.float32)
        }
        
        self.trim_mel       = trim_mel
        self.trim_factor    = tf.cast(trim_factor, tf.float32)
        self.trim_mel_method    = trim_mel_method if trim_mel else None
        
        super().init_train_config(** kwargs)
        
    def _build_model(self, architecture_name, ** kwargs):
        super()._build_model(
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
    def text_encoder_file(self):
        return os.path.join(self.save_dir, 'text_encoder.json')
    
    @property
    def mel_fn_file(self):
        return os.path.join(self.save_dir, 'mel_fn.json')
    
    @property
    def mel_input_shape(self):
        mel_length = self.max_input_length if self.use_fixed_length_input else None
        if not self.mel_as_image:
            return (mel_length, self.n_mel_channels)
        else:
            return (mel_length, self.n_mel_channels, 1)
    
    @property
    def decoder_method(self):
        if not self.use_ctc_decoder: return 'greedy'
        return 'beam_search'
    
    @property
    def input_signature(self):
        mel_input = (
            tf.TensorSpec(shape = (None,) + self.mel_input_shape, dtype = tf.float32),
            tf.TensorSpec(shape = (None,), dtype = tf.int32)
        )
        
        if self.use_ctc_decoder:
            return mel_input
        
        return mel_input + (
            tf.TensorSpec(shape = (None, None),     dtype = tf.int32),
            tf.TensorSpec(shape = (None,),          dtype = tf.int32)
        )
        
    @property
    def output_signature(self):
        return (
            tf.TensorSpec(shape = (None, None), dtype = tf.int32),
            tf.TensorSpec(shape = (None,), dtype = tf.int32)
        )
        
    @property
    def training_hparams(self):
        return super().training_hparams(
            max_input_length    = None,
            max_output_length   = None,
            
            trim_audio   = False,
            reduce_noise = False,
            trim_threshold   = 0.1,
            max_silence  = 0.15,
            trim_method  = 'window',
            trim_mode    = 'start_end',

            trim_mel     = False,
            trim_factor  = 0.6,
            trim_mel_method  = 'max_start_end'
        )

    @property
    def audio_rate(self):
        return self.mel_fn.sampling_rate
    
    @property
    def n_mel_channels(self):
        return self.mel_fn.n_mel_channels
    
    @property
    def vocab(self):
        return self.text_encoder.vocab

    @property
    def vocab_size(self):
        return self.text_encoder.vocab_size
                
    @property
    def blank_token_idx(self):
        return self.text_encoder.blank_token_idx
    
    @property
    def sos_token_idx(self):
        return self.text_encoder.sos_token_idx
    
    @property
    def eos_token_idx(self):
        return self.text_encoder.eos_token_idx
    
    @property
    def blank_token_idx(self):
        return self.text_encoder.blank_token_idx
    
    @property
    def sep_token(self):
        return '-'
    
    def __str__(self):
        des = super().__str__()
        des += "Language : {}\n".format(self.lang)
        des += "Audio rate : {}\n".format(self.audio_rate)
        des += "Mel channels : {}\n".format(self.n_mel_channels)
        des += "Use CTC decoder : {}\n".format(self.use_ctc_decoder)
        des += "Output vocab (size = {}) : {}\n".format(self.vocab_size, self.vocab)
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
            output = self(inputs, training = training)
            return output if not decode else self.decode_output(output)

        output = self.stt_model.infer(inputs, training = training, ** kwargs)
        return output if not decode else (self.decode_output(output[0]), output[1])
    
    def compile(self, loss = None, metrics = None, ** kwargs):
        if loss is None:
            loss = 'CTCLoss' if self.use_ctc_decoder else 'TextLoss'
        if metrics is None:
            if self.use_ctc_decoder:
                metrics = [{'metric' : 'TextMetric', 'config': {'pad_value' : self.blank_token_idx}}]
            else:
                metrics = ['TextAccuracy']
            
        super().compile(loss = loss, metrics = metrics, ** kwargs)
    
    def encode_text(self, text):
        return self.text_encoder.encode(text)
    
    def decode_text(self, encoded):
        return self.text_encoder.decode(encoded)
    
    def decode_output(self, output, * args, ** kwargs):
        if len(output.shape) in (1, 2):
            return self.decode_text(output)
        elif len(output.shape) == 3:
            pred = decode(output, method = self.decoder_method, blank_idx = self.blank_token_idx, ** kwargs)
            if self.use_ctc_decoder:
                #pred = [np.array([k for k, _ in itertools.groupby(p)]) for p in pred]
                return [self.decode_output(p) for p in pred]
            else:
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
    
    def get_audio_input(self, data):
        return load_audio(
            data, self.audio_rate, min_factor = self.trim_factor, ** self.trim_kwargs
        )
        
    def get_mel_input(self, data):
        mel = load_mel(
            data, self.mel_fn, trim_mode = self.trim_mel_method,
            min_factor = self.trim_factor, ** self.trim_kwargs
        )

        if len(mel.shape) == 3: mel = tf.squeeze(mel, 0)
        if self.mel_as_image:
            mel = tf.expand_dims(mel, axis = -1)
        
        return mel
    
    def encode_data(self, data):
        encoded_text = tf.py_function(self.encode_text, [data['text']], Tout = tf.int32)
        encoded_text.set_shape([None])
        
        mel = self.get_mel_input(data)
        
        return mel, len(mel), encoded_text, len(encoded_text)
    
    def filter_data(self, mel, mel_length, text, text_length):
        return tf.logical_and(
            mel_length <= self.max_input_length, 
            text_length <= self.max_output_length
        )
        
    def augment_data(self, mel, mel_length, text, text_length):
        maxval = self.max_input_length - mel_length
        if maxval > 0:
            padding_left = tf.random.uniform(
                (), minval = 0, 
                maxval = maxval,
                dtype = tf.int32
            )
            
            if maxval - padding_left > 0:
                padding_right = tf.random.uniform(
                    (), minval = 0, 
                    maxval = maxval - padding_left,
                    dtype = tf.int32
                )
            else:
                padding_right = 0
            
            if self.mel_as_image:
                padding = [(padding_left, padding_right), (0, 0), (0, 0)]
            else:
                padding = [(padding_left, padding_right), (0, 0)]
            
            mel = tf.pad(mel, padding)
        
        mel = tf.cond(
            tf.random.uniform(()) < self.augment_prct,
            lambda: mel + tf.random.uniform(
                tf.shape(mel), minval = -1., maxval = 1., dtype = mel.dtype
            ),
            lambda: mel
        )
        return mel, len(mel), text, text_length
        
    def preprocess_data(self, mel, mel_length, text, text_length):
        if self.use_ctc_decoder:
            return (mel, mel_length), (text, text_length)
        
        return (mel, mel_length, text[:, 1:], text_length - 1), (text[:, :-1], text_length - 1)
        
    def get_dataset_config(self, ** kwargs):
        kwargs['pad_kwargs']    = {
            'padding_values' : (self.pad_mel_value, 0, self.blank_token_idx, 0)
        }
        if self.use_fixed_length_input:
            kwargs['pad_kwargs']['padded_shapes'] = (
                self.mel_input_shape, (), (None,), ()
            )
        kwargs['batch_before_map']  = True
        kwargs['padded_batch']      = True
        
        return super().get_dataset_config(** kwargs)
        
    def train_step(self, batch):
        inputs, target = batch
        
        with tf.GradientTape() as tape:
            pred = self(inputs, training = True)
            if not self.use_ctc_decoder: pred, _ = pred
            
            loss = self.stt_model_loss(target, pred)
        
        variables = self.stt_model.trainable_variables

        grads = tape.gradient(loss, variables)
        self.stt_model_optimizer.apply_gradients(zip(grads, variables))
        
        return self.update_metrics(target, pred)
        
    def eval_step(self, batch):
        inputs, target = batch

        pred = self(inputs, training = False)
        if not self.use_ctc_decoder: pred, _ = pred
        
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
            pred, _ = self(inputs, training = False)
            infer, _ = self.infer(
                inputs[:2] if not self.use_ctc_decoder else inputs,
                max_length      = max_length,
                early_stopping  = False
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
        logging.info(des)
    
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
                audio = self.get_audio_input(filename)
                time_logger.stop_timer('processing')
            else:
                audio = filename
                if save:
                    filename = 'audio_{}.wav'.format(len(os.listdir(audio_dir)))
                    write_audio(audio, filename, rate = self.audio_rate)
            
            logging.info("Processing file {}...".format(filename))
            
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
                self.get_mel_input(a) for a in inputs if len(a) > self.audio_rate * MIN_AUDIO_TIME
            ]
            time_logger.stop_timer('processing')
            
            # Get text result 
            text_outputs = []
            for b in tqdm(range(0, len(inputs), batch_size)):
                mels    = inputs[b : b + batch_size]
                length  = tf.cast([len(mel) for mel in mels], tf.int32)
                
                batch   = [pad_batch(mels, pad_value = self.pad_mel_value), length]
                pred    = self.infer(batch, decode = True)
                if not self.use_ctc_decoder: pred, _ = pred
                
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
    
    def stream(self, max_time = 30, filename = None):
        import sounddevice as sd
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
        
        if filename is not None: write_audio(audio, filename, rate = self.audio_rate)
    
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
        config['lang']              = self.lang
        config['pad_mel_value']     = self.pad_mel_value
        config['mel_as_image']      = self.mel_as_image
        config['use_ctc_decoder']   = self.use_ctc_decoder
        
        config['mel_fn_type']       = self.mel_fn_file.replace(os.path.sep, '/')
        config['text_encoder']      = self.text_encoder_file.replace(os.path.sep, '/')
        config['use_fixed_length_input']    = self.use_fixed_length_input
        
        return config
