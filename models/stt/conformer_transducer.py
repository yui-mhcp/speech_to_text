
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

import logging
import itertools
import numpy as np
import tensorflow as tf

from loggers import timer
from utils import Producer
from utils.audio import load_audio, ConformerSTFT
from utils.text import SentencePieceTextEncoder
from custom_architectures.transformers_arch import conformer_arch
from custom_architectures.transducer_generation_utils import TransducerInferenceOutput
from models.stt.base_stt import BaseSTT

time_logger = logging.getLogger('timer')

def get_transduction_tokens(tokens, blank_mask):
    if len(tokens.shape) == 2:
        return [
            get_transduction_tokens(tok, b_mask) for tok, b_mask in zip(tokens, blank_mask)
        ]
    return tf.boolean_mask(tokens, tf.cast(1. - blank_mask, tf.bool))

def decode_transduction(output, blank_index):
    if len(output.shape) == 3: output = tf.expand_dims(output, axis = 0)
    if tf.shape(output)[1] > 0 and tf.shape(output)[2] > 0: output = tf.argmax(output, axis = -1)
    output = output.numpy()
    
    batch_size  = output.shape[0]
    
    tokens = []
    for b in range(len(output)):
        tokens_b, f_idx, t_idx = [], 0, 0
        while f_idx < output.shape[1] and t_idx < output.shape[2]:
            tok = output[b, f_idx, t_idx]
            if tok == blank_index:
                f_idx += 1
            else:
                t_idx += 1
                tokens_b.append(tok)
        tokens.append(np.array(tokens_b))
    return tokens

class ConformerTransducer(BaseSTT):
    def __init__(self, * args, ** kwargs):
        kwargs.update({
            'audio_format'      : 'mel',
            'use_ctc_decoder'   : False,
            'architecture_name' : 'ConformerTransducer'
        })
        super(ConformerTransducer, self).__init__(* args, ** kwargs)

    def _build_model(self, pretrained = None, ** kwargs):
        if pretrained is not None and not isinstance(pretrained, str):
            super(BaseSTT, self)._build_model(
                stt_model = conformer_arch.ConformerTransducer.from_pretrained(
                    pretrained = pretrained, ** kwargs
                )
            )
        else:
            kwargs['architecture_name'] = 'ConformerTransducer'
            super(ConformerTransducer, self)._build_model(** kwargs)

    def compile(self, ** kwargs):
        kwargs.update({
            'loss'  : 'RNNTLoss',
            'metrics'   : [],
            'loss_config'   : {'blank_index' : self.blank_token_idx}
        })
            
        super().compile(** kwargs)

    def decode_output(self, output, * args, ** kwargs):
        if isinstance(output, TransducerInferenceOutput):
            tokens = get_transduction_tokens(
                output.tokens, output.blank_mask
            )
        elif len(output.shape) in (3, 4):
            tokens = decode_transduction(output, self.blank_token_idx)

        return self.decode_text(tokens)

    def preprocess_data(self, mel, mel_length, text, text_length):
        if self.use_ctc_decoder:
            return (mel, mel_length), (text, text_length)
        
        text_in, text_out = text, text
        text_in_len, text_out_len   = text_length, text_length
        if self.text_encoder.use_sos_and_eos:
            text_in, text_in_len    = text_in[:, :-1], text_in_len - 1
            text_out, text_out_len  = text_out[:, 1:], text_out_len - 1
        
        sos = tf.tile(tf.reshape(self.blank_token_idx, [1, 1]), [tf.shape(text_in)[0], 1])
        text_in = tf.concat([sos, text_in], axis = 1)
        
        return (mel, mel_length, text_in, text_in_len), (text_out, text_out_len)

    @timer
    def stream(self,
               stream   = None,
               playback = False,
               
               processing_fn    = None,
               
               max_time = 10,
               time_window  = 0.25,
               
               max_frames   = 10000,
               
               n_succ       = 1,
               
               max_workers  = 0,
               
               restart_after_succ   = -1,
               
               ** kwargs
              ):
        @timer
        def get_length(blank_mask):
            changes = tf.where(blank_mask[0] == 0.)
            if len(changes) == 0: return 0, -1
            
            last_change = changes[-1, 0]
            return last_change, tf.cast(tf.reduce_sum(blank_mask[0, : last_change]), tf.int32) + 1
        
        def get_tokens(tokens):
            if len(tf.shape(tokens)) == 2: tokens = tokens[0]
            return [int(t) for t, _ in itertools.groupby(tokens.numpy()) if t != self.stt_model.hparams.pad_token]

        def microphone_audio_stream():
            p = pyaudio.PyAudio()
            inp_stream = p.open(
                format = pyaudio.paFloat32, channels = 1, rate = self.audio_rate, input = True
            )

            print('Start recording...')
            for i in range(int(max_time / time_window + 1)):
                yield np.frombuffer(inp_stream.read(frames_per_buffer), dtype = np.float32)
            yield None

            print('\nStop recording !')
            inp_stream.close()

        def file_audio_stream():
            audio = load_audio(stream, self.audio_rate)
            
            end = min(int(max_time * self.audio_rate), len(audio))
            for start in range(0, end, frames_per_buffer):
                yield audio[start : start + frames_per_buffer]
            yield None

        @timer
        def audio_to_mel(audio_frame, audio = None):
            if audio_frame is None: return None, (audio, )
            audio = audio_frame if audio is None else tf.concat([audio, audio_frame], axis = 0)
            return tf.expand_dims(self.get_audio(audio), axis = 0), (audio, )

        @timer
        def encode(mel):
            if mel is None: return None, ()
            
            if not isinstance(mel, (list, tuple)) and tf.shape(mel)[1] > max_frames:
                print('# frames : {}'.format(mel.shape))
                mel = mel[:, - max_frames :]

            encoder_output = self.stt_model.encoder(
                mel, training = False
            )
            if isinstance(encoder_output, (list, tuple)): encoder_output = encoder_output[0]

            return encoder_output, ()

        @timer
        def decode(encoder_output, last_tokens = None, mask = None, model_state = None, state_idx = 0, prev_outputs = None):
            if encoder_output is None or tf.shape(encoder_output)[1] == 0:
                return None, (last_tokens, mask, model_state, state_idx)

            output = self.stt_model.infer(
                tokens = last_tokens,
                blank_mask = mask,
                encoder_output = encoder_output[:, state_idx :],

                #restart_after_succ = restart_after_succ,

                initial_state   = model_state,
                return_state = True,
                ** kwargs
            )

            if prev_outputs is None: prev_outputs = []
            prev_outputs.append((output, get_tokens(output.tokens)))
            
            time_logger.start_timer('compare outputs')
            
            stable_output, stable_toks = None, None
            if len(prev_outputs) > n_succ:
                stable_idx = 0
                for i, (out, toks) in enumerate(prev_outputs[: - n_succ]):
                    n_stable = 0
                    for j, (succ_out, succ_toks) in enumerate(prev_outputs[i :]):
                        if toks == succ_toks[:len(toks)]:
                            n_stable += 1
                    
                    if n_stable >= n_succ:
                        stable_output, stable_toks = out, toks
                        stable_idx    = i
                    
                if stable_idx > 0:
                    prev_outputs = [
                        (out, toks) for out, toks in prev_outputs[stable_idx + 1 :]
                        if toks[:len(stable_toks)] == stable_toks
                    ]
                
            
            time_logger.stop_timer('compare outputs')
            last_idx = 0
            if stable_output is not None:
                last_idx, n_frames = get_length(stable_output.blank_mask)
            
            if last_idx > 0:
                last_idx += 2
                next_state = (stable_output.tokens[..., :last_idx], stable_output.blank_mask[..., :last_idx], stable_output.state, n_frames)
            else:
                next_state = (last_tokens, mask, model_state, state_idx)

            unstable_toks = None if len(prev_outputs) == 0 else prev_outputs[-1][1]
            return (stable_toks, unstable_toks), next_state + (prev_outputs, )

        @timer
        def show_output(output, prev = None):
            if output is None:
                print()
                return None, ()

            stable, unstable = output
            if stable is None: stable = prev
            
            if stable is not None and unstable: unstable = unstable[len(stable) :]
            
            text = self.decode_text(stable) if stable else ''
            unstable_text = self.decode_text(unstable) if unstable else ''
            if text or unstable_text:
                show = '{}[{}]'.format(text, unstable_text)
                print(show + ' ' * (250 - len(text)), end = '\n')
            return text, (stable, )

        if processing_fn is None: processing_fn = audio_to_mel
        
        frames_per_buffer = int(time_window * self.audio_rate)
        
        if playback:
            p = pyaudio.PyAudio()

            out_stream = p.open(
                format = pyaudio.paFloat32, channels = 1, rate = self.audio_rate, output = True,
                frames_per_buffer = frames_per_buffer
            )
        
        prod = Producer(
            microphone_audio_stream if stream is None or isinstance(stream, int) else file_audio_stream,
            run_main_thread = True if max_workers < 0 else False
        )
        
        pipe = prod.add_consumer([
            {'consumer' : fn, 'stateful' : True}
            for fn in [processing_fn, encode, decode, show_output]
        ], link_stop = True, start = True, max_workers = max_workers)
        
        if playback:
            play_cons = prod.add_consumer(
                lambda frame: out_stream.write(frame.tobytes()) if frame is not None else None,
                start = True,
                link_stop   = True,
                max_workers = min(max_workers, 0),
                stop_listeners  = out_stream.close
            )

        prod.start()
        prod.join(recursive = True)

    @classmethod
    def from_nemo_pretrained(cls,
                             pretrained_name,
                             nom     = None,
                             lang    = 'en',
                             pretrained = None,
                             ** kwargs
                            ):
        if nom is None: nom = pretrained_name
        
        if pretrained is None:
            import nemo.collections.asr as nemo_asr
            pretrained = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(pretrained_name)

        pretrained_stft = pretrained.preprocessor
        
        mel_fn = ConformerSTFT(
            n_mel_channels  = pretrained.cfg.preprocessor.features,
            sampling_rate   = pretrained.cfg.preprocessor.sample_rate,
            win_length      = pretrained_stft.featurizer.win_length,
            hop_length      = pretrained_stft.featurizer.hop_length,
            normalize_mode  = pretrained.cfg.preprocessor.normalize,
            filter_length   = pretrained_stft.featurizer.n_fft,
            window      = pretrained.cfg.preprocessor.window,
            pre_emph    = pretrained_stft.featurizer.preemph,
            mag_power   = pretrained_stft.featurizer.mag_power,
            log_zero_guard_value    = pretrained_stft.featurizer.log_zero_guard_value,
            periodic    = False,
            log     = True
        )
        
        vocab = pretrained.decoding.tokenizer.vocab
        if pretrained.decoding.tokenizer.pad_id == -1:
            vocab.append('<pad>')
        text_encoder = SentencePieceTextEncoder(
            vocab       = pretrained.decoding.tokenizer.vocab,
            tokenizer   = pretrained.decoding.tokenizer.tokenizer,
            pad_token   = '<pad>'
        )
        
        instance = cls(
            nom     = nom,
            lang    = lang,
            mel_fn  = mel_fn, 
            text_encoder = text_encoder,

            pretrained  = pretrained,
            max_to_keep  = 1,
            pretrained_name = 'nemo_{}'.format(pretrained_name),
            ** kwargs
        )
        
        instance.save()
        
        return instance
