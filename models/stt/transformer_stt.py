
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

from models.stt.base_stt import BaseSTT

class TransformerSTT(BaseSTT):
    def __init__(self, * args, ** kwargs):
        kwargs.update({
            'audio_format'      : 'mel',
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
