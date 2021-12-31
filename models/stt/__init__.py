import os

from models.model_utils import get_model_config

from models.stt.base_stt import BaseSTT
from models.stt.jasper import Jasper
from models.stt.deep_speech import DeepSpeech
from models.stt.transformer_stt import TransformerSTT

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
    return model.search(keyword, audios, ** kwargs)

def predict(audios, lang = None, model = None, ** kwargs):
    model = get_model(lang = lang, model = model)
    return model.predict(audios, ** kwargs)


_pretrained = {
    'en'    : 'pretrained_jasper'
}

_models = {
    'BaseSTT'   : BaseSTT,
    'Jasper'    : Jasper,
    'DeepSpeech'    : DeepSpeech,
    'TransformerSTT'    : TransformerSTT
}