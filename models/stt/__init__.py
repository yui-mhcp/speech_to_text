import os

from models.stt.base_stt import BaseSTT
from models.stt.jasper import Jasper
from models.stt.deep_speech import DeepSpeech
from models.stt.transformer_stt import TransformerSTT

from utils.generic_utils import load_json

def get_model(lang = None, model = None, ** kwargs):
    assert lang is not None or model is not None
    if model is None:
        if lang not in _pretrained:
            raise ValueError("No pretrained model for this language !!\n  Supported : {}\n   Got : {}".format(list(_pretrained.keys()), lang))
        
        model = _pretrained[lang]
    
    if isinstance(model, str):
        config_file = os.path.join('pretrained_models', model, 'config.json')
        model_class = load_json(config_file).get('class_name', None)
        
        if model_class is None or model_class not in _models:
            raise ValueError("Unknown model class : !\n  Accepted : {}\n  Got : {}".format(list(_models.keys()), model_class))
        
        model = _models[model_class](nom = model)
    
    return model

def search(keyword, audios, lang = None, model = None, ** kwargs):
    model = get_model(lang, model = model)
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