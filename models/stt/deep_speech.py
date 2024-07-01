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

import tensorflow as tf

from utils.text import TextEncoder
from utils.audio import DeepSpeechSTFT
from custom_architectures import get_architecture
from models.weights_converter import partial_transfer_learning
from models.stt.base_stt import BaseSTT, _deep_speech_en_symbols

class DeepSpeech(BaseSTT):
    def __init__(self, * args, ** kwargs):
        kwargs.update({
            'audio_format'  : 'mel',
            'architecture'  : 'deep_speech_2'
        })
        super().__init__(* args, ** kwargs)

    @classmethod
    def from_deep_speech_pretrained(cls, 
                                    nom      = 'pretrained_deep_speech',
                                    lang     = 'en',
                                    vocab    = _deep_speech_en_symbols, 
                                    ** kwargs
                                   ):
        mel_fn = DeepSpeechSTFT(
            sampling_rate   = 16000,
            n_mel_channels  = 160,
            filter_length   = 0.02,
            hop_length      = 0.01,
            win_length      = 0.02,
            normalize_mode  = 'per_feature'
        )
        
        text_encoder = TextEncoder(
            vocab           = vocab, 
            level           = 'char',
            vocab_size      = kwargs.pop('vocab_size', None),
            cleaners        = kwargs.pop('cleaners', ['english_cleaners']),
            ukn_token       = None, 
            use_sos_and_eos = False, 
            name            = 'DeepSpeech text encoder'
        )
            
        instance = cls(
            nom     = nom,
            lang    = lang,
            mel_fn  = mel_fn, 
            text_encoder = text_encoder,
            
            max_to_keep  = 1,
            pretrained_name = 'pretrained_deep_speech',
            ** kwargs
        )
        
        with tf.device('cpu') as dev:
            pretrained_model = get_architecture(
                'DeepSpeech2', input_shape = (None, 160), vocab_size = 29, pretrained = True
            )
        
        partial_transfer_learning(instance.stt_model, pretrained_model)
        
        instance.save()
        
        return instance
