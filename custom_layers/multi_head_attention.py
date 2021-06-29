import tensorflow as tf

from hparams.hparams import HParams

HParamsMHA = HParams(
    num_heads   = 8,
    attention_dim   = 512,
    attention_drop_rate    = 0.,
                 
    drop_rate   = 0.1,
    normalize   = True,
    epsilon     = 1e-6,
    use_output_layer    = True
)

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, name = None, ** kwargs):
        super().__init__(name = name)
        self.hparams = HParamsMHA.extract(kwargs)
        
        assert self.hparams.attention_dim % self.hparams.num_heads == 0, "Attention_dim % num_heads != 0 !"
        
        self.num_heads = self.hparams.num_heads
        self.attention_dim  = self.hparams.attention_dim
        self.depth = self.hparams.attention_dim // self.hparams.num_heads
        
        self.wq = tf.keras.layers.Dense(self.hparams.attention_dim, name = "query_layer")
        self.wk = tf.keras.layers.Dense(self.hparams.attention_dim, name = "key_layer")
        self.wv = tf.keras.layers.Dense(self.hparams.attention_dim, name = "value_layer")
        
        self.attn_dropout   = tf.keras.layers.Dropout(self.hparams.attention_drop_rate) if self.hparams.attention_drop_rate > 0. else None

        self.output_layer   = tf.keras.layers.Dense(self.hparams.attention_dim) if self.hparams.use_output_layer else None
        self.dropout        = tf.keras.layers.Dropout(self.hparams.drop_rate)
        self.norm_layer     = tf.keras.layers.LayerNormalization() if self.hparams.normalize else None
        
    def split_heads(self, x, batch_size):
        """
            Split the last dimension into (num_heads, depth)
            Transpose the result such that the shape is (batch, num_heads, seq_len, depth)
        """
        
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm = [0, 2, 1, 3])
    
    def scaled_dot_product_attention(self, q, k, v, mask = None, training = False):
        """
            Attention(Q, K, T) = softmax(Q @ K^t / sqrt(d_k)) * V

            Arguments :
                - q : query shape == (..., seq_len_q, depth)
                - k : key shape == (..., seq_len_k, depth)
                - v : value shape == (..., seq_len_v, depth_v)
            Outputs : output, attention_weights
                - attention_weights shape == (..., seq_len_q, seq_len_k)
                - output shape == (..., seq_len_q, depth_v)
        """
        matmul_qk = tf.matmul(q, k, transpose_b = True)

        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        # (..., seq_len_q, seq_len_k)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis = -1)

        if self.attn_dropout is not None:
            attention_weights = self.attn_dropout(attention_weights, training = training)
        # (..., seq_len_q, depth_v)
        output = tf.matmul(attention_weights, v) 

        return output, attention_weights
    
    def call(self, query, key, value, mask = None, training = False, return_attention = True):        
        batch_size = tf.shape(query)[0]
        
        q = self.wq(query)      # (batch_size, seq_len, d_model)
        k = self.wk(key)        # (batch_size, seq_len, d_model)
        v = self.wv(value)      # (batch_size, seq_len, d_model)
        
        q = self.split_heads(q, batch_size)     # (batch, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)     # (batch, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)     # (batch, num_heads, seq_len_v, depth)
                
        # scaled_attention shape == (atch, num_heads, seq_len_q, depth)
        # attention_weights shape == (batch, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, mask, training)
        
        # batch, seq_len_q, num_heads, depth
        scaled_attention = tf.transpose(scaled_attention, perm = [0, 2, 1, 3])
        
        # (batch, seq_len_q, d_model)
        output = tf.reshape(scaled_attention, (batch_size, -1, self.attention_dim))
        
        if self.output_layer is not None:   output = self.output_layer(output)
        if self.dropout is not None:        output = self.dropout(output, training = training)
        if self.norm_layer is not None:     output = self.norm_layer(output, training = training)
        
        return output, attention_weights if return_attention else output
    
    def get_config(self):
        config = super().get_config()
        return self.hparams + config
        
       
def scaled_dot_product_attention(q, k, v, mask = None):
    """
        Attention(Q, K, T) = softmax(Q @ K^t / sqrt(d_k)) * V
    
        Arguments :
            - q : query shape == (..., seq_len_q, depth)
            - k : key shape == (..., seq_len_k, depth)
            - v : value shape == (..., seq_len_v, depth_v)
        Outputs : output, attention_weights
            - attention_weights shape == (..., seq_len_q, seq_len_k)
            - output shape == (..., seq_len_q, depth_v)
    """
    matmul_qk = tf.matmul(q, k, transpose_b = True)
    
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
      
    # (..., seq_len_q, seq_len_k)
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis = -1)
    
    # (..., seq_len_q, depth_v)
    output = tf.matmul(attention_weights, v) 
    
    return output, attention_weights

class Conv2DMultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, n = 64, c = 64, kernel_initializer = 'glorot_normal', **kwargs):
        super(Conv2DMultiHeadAttention, self).__init__(**kwargs)
        self.n = n
        self.c = c
        
        
        self.conv_q = tf.keras.layers.Conv2D(filters = c, kernel_size = 3, 
                                             padding = 'same', name = "conv_q",
                                             kernel_initializer = kernel_initializer)
        self.conv_k = tf.keras.layers.Conv2D(filters = c, kernel_size = 3, 
                                             padding = 'same', name = "conv_k",
                                             kernel_initializer = kernel_initializer)
        self.conv_v = tf.keras.layers.Conv2D(filters = c, kernel_size = 3, 
                                             padding = 'same', name = "conv_v",
                                             kernel_initializer = kernel_initializer)
        
        self.conv = tf.keras.layers.Conv2D(filters = n, kernel_size = 3, padding = 'same', 
                                           kernel_initializer = kernel_initializer, 
                                           name = "conv")
        
        self.norm_q = tf.keras.layers.BatchNormalization()
        self.norm_k = tf.keras.layers.BatchNormalization()
        self.norm_v = tf.keras.layers.BatchNormalization()
        self.norm = tf.keras.layers.LayerNormalization()
        
        self.final_conv_1 = tf.keras.layers.Conv2D(filters = n, kernel_size = 3, 
                                                   padding = 'same', activation = 'relu',
                                                   kernel_initializer = kernel_initializer, 
                                                   name = "final_conv_1")
        self.final_conv_2 = tf.keras.layers.Conv2D(filters = n, kernel_size = 3, 
                                                   padding = 'same',
                                                   kernel_initializer = kernel_initializer, 
                                                   name = "final_conv_2")
        
        self.final_norm_1 = tf.keras.layers.BatchNormalization()
        self.final_norm_2 = tf.keras.layers.BatchNormalization()
        self.act = tf.keras.layers.Activation('relu')
            
    def call(self, x):
        batch_size = tf.shape(x)[0]
        
        q = self.norm_q(self.conv_q(x))
        k = self.norm_k(self.conv_k(x))
        v = self.norm_v(self.conv_v(x))
        
        q_time = tf.transpose(q, [0, 3, 1, 2])
        k_time = tf.transpose(k, [0, 3, 1, 2])
        v_time = tf.transpose(v, [0, 3, 1, 2])
        
        q_fre = tf.transpose(q, [0, 3, 2, 1])
        k_fre = tf.transpose(k, [0, 3, 2, 1])
        v_fre = tf.transpose(v, [0, 3, 2, 1])
        
                   
        scaled_attention_time, attention_weights_time = scaled_dot_product_attention(
            q_time, k_time, v_time, None
        )
        scaled_attention_fre, attention_weights_fre = scaled_dot_product_attention(
            q_fre, k_fre, v_fre, None
        )
        

        scaled_attention_time = tf.transpose(scaled_attention_time, [0, 2, 3, 1])
        
        scaled_attention_fre = tf.transpose(scaled_attention_fre, [0, 3, 2, 1])
        

        concat_attention = tf.concat([scaled_attention_time, scaled_attention_fre], 
                                      axis = -1)
        

        output = self.norm(self.conv(concat_attention) + x)
        
        final_output = self.final_norm_1(self.final_conv_1(output))
        final_output = self.final_norm_2(self.final_conv_2(final_output))
        
        final_output = self.act(final_output + output)
        
        return final_output, attention_weights_time, attention_weights_fre
    
    def get_config(self):
        config = super(Conv2DMultiHeadAttention, self).get_config()
        config['n'] = self.n
        config['c'] = self.c
        return config

