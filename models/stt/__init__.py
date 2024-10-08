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

from models.utils import get_model_config

from .base_stt import BaseSTT
from .jasper import Jasper
from .deep_speech import DeepSpeech
from .transformer_stt import TransformerSTT
from .whisper import Whisper

def get_model_lang(model):
    return get_model_config(model).get('lang', None)

def get_model_name(lang):
    if lang not in _pretrained:
        raise ValueError('Unknown language for pretrained TTS model\n  Accepted : {}\n  Got : {}'.format(tuple(_pretrained.keys()), lang))
    return _pretrained[lang]

def get_model(lang = None, model = None, ** kwargs):
    if model is None:
        assert lang is not None, "You must specify either the model or the language !"
        model = get_model_name(lang)
    
    if isinstance(model, str):
        from models import get_pretrained
        model = get_pretrained(model)
    
    return model

def search(keyword, audios, lang = None, model = None, ** kwargs):
    model = get_model(lang = lang, model = model)
    return model.search(keyword, audios, lang = lang, ** kwargs)

def predict(audios, lang = None, model = None, ** kwargs):
    model = get_model(lang = lang, model = model)
    return model.predict(audios, lang = lang, ** kwargs)


_pretrained = {
    'en'    : 'pretrained_jasper'
}

