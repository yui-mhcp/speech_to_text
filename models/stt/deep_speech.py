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
        return super().call(inputs[0], training = training)
