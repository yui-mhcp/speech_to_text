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

import time
import keras
import unittest
import threading
import numpy as np
import keras.ops as K

from absl.testing import parameterized

from . import CustomTestCase, is_tensorflow_available
from utils.keras import TensorSpec, ops, graph_compile, execute_eagerly, compile_function
from utils.keras.ops.execution_contexts import XLAExecution

class TestExecuteEagerly(CustomTestCase, parameterized.TestCase):
    @parameterized.parameters(False, True)
    def test_simple(self, numpy):
        @execute_eagerly(Tout = 'float32', numpy = numpy)
        def foo(x, y):
            self.assertTrue(ops.executing_eagerly())
            if not numpy:
                self.assertTensor(x)
            else:
                self.assertArray(x)
            self.assertTrue(isinstance(y, float))
            return x * y
        
        foo(ops.arange(5, dtype = 'float32'), y = 2.)

    @unittest.skipIf(not is_tensorflow_available(), 'Tensorflow is not available')
    @parameterized.parameters(False, True)
    def test_tensorflow_simple(self, numpy):
        import tensorflow as tf
        
        @tf.function
        def foo(x, y):
            self.assertFalse(tf.executing_eagerly())
            self.assertTrue(ops.is_tensorflow_graph())
            out = foo_eager(x, y)
            self.assertTrue(out.dtype == tf.float32, 'Dtype is {}'.format(out.dtype))
            self.assertTrue(out.shape.rank is None, 'The shape should be unknown but is {}'.format(out.shape))
            return out
        
        @execute_eagerly(Tout = 'float32', numpy = numpy)
        def foo_eager(x, y):
            self.assertTrue(tf.executing_eagerly())
            self.assertTrue(ops.executing_eagerly())
            self.assertFalse(ops.is_tensorflow_graph())
            if not numpy:
                self.assertTfTensor(x)
                self.assertTfTensor(y)
            else:
                self.assertArray(x)
                self.assertArray(y)
            return tf.cast(x, 'float32') * y
        
        foo(tf.range(5), tf.constant(2, 'float32'))
    
    @unittest.skipIf(not is_tensorflow_available(), 'Tensorflow is not available')
    @parameterized.parameters(False, True)
    def test_tensorflow_with_kwargs(self, numpy):
        import tensorflow as tf
        
        @tf.function
        def foo(x, y):
            self.assertFalse(tf.executing_eagerly())
            out = foo_eager(x, y = y)
            self.assertTrue(out.dtype == tf.float32, 'Dtype is {}'.format(out.dtype))
            self.assertTrue(out.shape.rank is None, 'The shape should be unknown but is {}'.format(out.shape))
            return out
        
        @execute_eagerly(Tout = 'float32', numpy = numpy)
        def foo_eager(x, *, y = None):
            self.assertTrue(tf.executing_eagerly())
            self.assertTrue(ops.executing_eagerly())
            self.assertTrue(y is not None, 'y has not been forwarded correctly')
            if not numpy:
                self.assertTfTensor(x)
                self.assertTfTensor(y)
            else:
                self.assertArray(x)
                self.assertArray(y)
            return tf.cast(x, 'float32') * y
        
        foo(tf.range(5), tf.constant(2, 'float32'))

    @unittest.skipIf(not is_tensorflow_available(), 'Tensorflow is not available')
    @parameterized.parameters(False, True)
    def test_tensorflow_with_signature(self, numpy):
        import tensorflow as tf
        
        @tf.function
        def foo(x, y):
            self.assertFalse(tf.executing_eagerly())
            out = foo_eager(x, y)
            self.assertTrue(isinstance(out, (list, tuple)), 'Got type {}'.format(type(out)))
            self.assertTrue(out[0].dtype == tf.float32, 'The dtype is {}'.format(out[0].dtype))
            self.assertTrue(len(out[0].shape) == 1, 'The shape is {}'.format(out[0].shape))

            self.assertTrue(out[1].dtype == tf.int32, 'The dtype is {}'.format(out[0].dtype))
            self.assertTrue(len(out[1].shape) == 0, 'The shape is {}'.format(out[1].shape))
            return out
        
        @execute_eagerly(signature = [
            TensorSpec(shape = (None, ), dtype = 'float32'),
            TensorSpec(shape = (), dtype = 'int32')
        ], numpy = numpy)
        def foo_eager(x, y):
            self.assertTrue(tf.executing_eagerly())
            self.assertTrue(ops.executing_eagerly())
            self.assertTrue(y is not None, 'y has not been forwarded correctly')
            if not numpy:
                self.assertTfTensor(x)
                self.assertTfTensor(y)
            else:
                self.assertArray(x)
                self.assertArray(y)
            return tf.cast(x, 'float32') * y, np.array(len(x), dtype = 'int32')
        
        foo(tf.range(5), tf.constant(2, 'float32'))

    def test_executing_eagerly(self):
        def foo(x):
            if ops.is_tensorflow_backend():
                import tensorflow as tf
                self.assertFalse(tf.executing_eagerly())
                self.assertTrue(ops.is_tensorflow_graph())
            self.assertFalse(ops.executing_eagerly())
            return x ** 2
        
        xla_foo = compile_function(foo, jit_compile = True)
        self.assertTrue(ops.executing_eagerly())
        with XLAExecution():
            # The context manager has no effect in tensorflow backend
            if ops.is_tensorflow_backend():
                self.assertTrue(ops.executing_eagerly())
            else:
                self.assertFalse(ops.executing_eagerly())
            xla_foo(2)
        
        self.assertTrue(ops.executing_eagerly())

class TestXLACompilation(CustomTestCase):
    def test_run_eagerly(self):
        @graph_compile
        def foo(x : TensorSpec(), eager = False):
            if eager:
                self.assertTrue(ops.executing_eagerly(), 'This should be executed eagerly')
            else:
                self.assertFalse(ops.executing_eagerly(), 'This should be executed in graph')
        
        self.assertTrue(ops.executing_eagerly())
        foo(2)
        self.assertTrue(ops.executing_eagerly())
        foo(2, True, run_eagerly = True)
    
    def test_run_eagerly_nested(self):
        @graph_compile
        def foo(x : TensorSpec(), eager = False):
            nested_foo(eager)

        @graph_compile
        def nested_foo(eager):
            if eager:
                self.assertTrue(ops.executing_eagerly(), 'This should be executed eagerly')
            else:
                self.assertFalse(ops.executing_eagerly(), 'This should be executed in graph')
        
        foo(2)
        self.assertTrue(ops.executing_eagerly())
        foo(2, True, run_eagerly = True)

    def test_multi_threading(self):
        @graph_compile
        def foo_eager():
            self.assertTrue(ops.executing_eagerly())
            time.sleep(0.01)
            self.assertTrue(ops.executing_eagerly())

        @graph_compile
        def foo():
            self.assertFalse(ops.executing_eagerly(), 'The separation between threads of `run_eagerly` seems not working')
        
        for _ in range(5):
            t = threading.Thread(
                target = foo_eager, kwargs = {'run_eagerly' : True, 'recompile' : True}
            )
            t.start()
            time.sleep(1e-3)

            foo(recompile = True)
            t.join()
        
    def test_simple_cast(self):
        @graph_compile
        def foo(x):
            self.assertTrue(isinstance(x, int))
        
        @graph_compile
        def foo_with_cast(x : TensorSpec()):
            self.assertTensor((x))
            self.assertTrue(ops.dtype_to_str(x.dtype), 'int32')
        
        @graph_compile
        def foo_with_spec(x : TensorSpec(dtype = 'float32')):
            self.assertTensor(x)
            self.assertTrue(ops.dtype_to_str(x.dtype), 'float32')
        
        foo(2)
        foo_with_cast(2)
        foo_with_cast(x = 2, recompile = True)
        foo_with_spec(2)
        foo_with_spec(x = 2, recompile = True)
    
    def test_cast_default(self):
        @graph_compile
        def foo(x = 2):
            self.assertTrue(isinstance(x, int))
        
        @graph_compile
        def foo_with_spec(x : TensorSpec(dtype = 'int32') = 2):
            self.assertTrue(ops.is_tensor(x))
            self.assertTrue(ops.dtype_to_str(x.dtype), 'int32')

        foo()
        foo_with_spec()
    
    def test_cast_kwargs(self):
        @graph_compile(kwargs_annots = {
            'y' : TensorSpec(), 'z' : TensorSpec(dtype = 'float32')
        })
        def foo(x, ** kwargs):
            self.assertTensor(kwargs['y'])
            self.assertTensor(kwargs['z'])
            self.assertTrue(ops.is_int(kwargs['y']), str(kwargs['y'].dtype))
            self.assertTrue(ops.is_float(kwargs['z']), str(kwargs['z'].dtype))
            self.assertTrue(isinstance(kwargs['method'], str))
            self.assertTrue(isinstance(kwargs['cond'], bool))
            self.assertTrue(isinstance(kwargs['weight'], float))
            self.assertTrue(kwargs['none'] is None)
        
        foo(2, y = 3, z = 4, method = 'default', cond = True, none = None, weight = 2.)
    
    @unittest.skipIf(
        keras.backend.backend() == 'tensorflow' or not is_tensorflow_available(),
        'This test requires `tensorflow` available while using another backend'
    )
    def test_force_tensorflow(self):
        import tensorflow as tf
        
        @graph_compile
        def foo(x):
            self.assertTrue(tf.executing_eagerly(), 'This should not detect the XLA execution of another backend')
            self.assertFalse(ops.executing_eagerly())

        @graph_compile(force_tensorflow = True)
        def foo_tf(x : TensorSpec(dtype = 'int32')):
            self.assertFalse(tf.executing_eagerly(), 'This should be executed in TF graph')
            self.assertTrue(ops.is_tensorflow_graph())
            self.assertFalse(ops.executing_eagerly(), 'The `ops.executing_eagerly` has to detect `tensorflow-graph` executions in all backends')
            self.assertTfTensor(x)
            self.assertEqual(ops.dtype_to_str(x.dtype), 'int32')
        
        foo(2)
        foo_tf(2)
