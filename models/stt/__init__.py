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
import importlib

from utils import setup_environment
from ..interfaces import BaseModel

for module in os.listdir(__package__.replace('.', os.path.sep)):
    if module.startswith(('.', '_')) or '_old' in module: continue
    module = importlib.import_module(__package__ + '.' + module[:-3])
    
    globals().update({
        k : v for k, v in vars(module).items() if isinstance(v, type) and issubclass(v, BaseModel)
    })

def get_model_name(lang):
    if lang not in _pretrained:
        raise ValueError('Unknown language for pretrained TTS model\n  Accepted : {}\n  Got : {}'.format(tuple(_pretrained.keys()), lang))
    return _pretrained[lang]

def get_model(lang = None, model = None):
    if model is None:
        assert lang is not None, "You must specify either the model or the language !"
        model = get_model_name(lang)
    
    if isinstance(model, (str, dict)):
        from models import get_pretrained
        
        model = get_pretrained(model)
    
    return model

def transcribe(audios, *, lang = None, model = None, ** kwargs):
    return get_model(lang = lang, model = model).predict(audios, lang = lang, ** kwargs)

def translate(audios, *, lang = None, model = None, ** kwargs):
    return get_model(lang = lang, model = model).predict(
        audios, lang = lang, task = 'translate', ** kwargs
    )

def stream(stream, *, lang = None, model = None, ** kwargs):
    setup_environment(** kwargs)
    return get_model(lang = lang, model = model).stream(stream, lang = lang, ** kwargs)

_pretrained = {}

