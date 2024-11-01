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

from utils import import_objects, limit_gpu_memory
from models.utils import get_model_config

globals().update(import_objects(
    __package__.replace('.', os.path.sep), allow_functions = False
))


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
        
        model = get_pretrained(model, ** kwargs)
    
    return model

def search(keyword, audios, lang = None, model = None, ** kwargs):
    return get_model(lang = lang, model = model, ** kwargs).search(
        keyword, audios, lang = lang, ** kwargs
    )

def transcribe(audios, lang = None, model = None, ** kwargs):
    return get_model(lang = lang, model = model, ** kwargs).predict(audios, lang = lang, ** kwargs)

def translate(audios, lang = None, model = None, ** kwargs):
    return get_model(lang = lang, model = model, ** kwargs).predict(
        audios, lang = lang, task = 'translate', ** kwargs
    )

def stream(stream, lang = None, model = None, ** kwargs):
    if 'gpu_memory' in kwargs: limit_gpu_memory(kwargs.pop('gpu_memory'))
    return get_model(lang = lang, model = model, ** kwargs).stream(stream, lang = lang, ** kwargs)


_pretrained = {}

