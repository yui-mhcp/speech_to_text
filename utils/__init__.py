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

from loggers import set_level

from .comparison_utils import is_diff, is_equal
from .distances import *
from .embeddings import *
from .file_utils import *
from .generic_utils import *
from .plot_utils import *
from .sequence_utils import *
from .wrappers import *

from .keras import *
from .callbacks import *
from .threading import *

def setup_environment(log_level = None, ** kwargs):
    if log_level: set_level(log_level)
    set_gpu_config(** kwargs)
