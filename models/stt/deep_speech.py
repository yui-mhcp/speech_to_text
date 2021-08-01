from models.stt.base_stt import BaseSTT

class DeepSpeech(BaseSTT):
    def __init__(self, * args, ** kwargs):
        kwargs.update({
            'mel_as_image'      : False,
            'use_ctc_decoder'   : True,
            'architecture_name' : 'deep_speech_2'
        })
        super().__init__(* args, ** kwargs)
    
    def call(self, inputs, training = False):
        if isinstance(inputs, (list, tuple)): inputs = inputs[0]
        return super().call(inputs, training = training)
