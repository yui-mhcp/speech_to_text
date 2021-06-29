from models.stt.base_stt import BaseSTT

class TransformerSTT(BaseSTT):
    def __init__(self, * args, ** kwargs):
        kwargs.update({
            'mel_as_image'      : False,
            'use_ctc_decoder'   : False
        })
        super().__init__(* args, ** kwargs)

    def _build_model(self, embedding_dim = 512, ** kwargs):
        super()._build_model(
            architecture_name   = 'transformer_stt',
            embedding_dim       = embedding_dim,
            sos_token   = self.sos_token_idx,
            eos_token   = self.sos_token_idx,
            ** kwargs
        )
