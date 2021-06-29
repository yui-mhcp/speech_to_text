import numpy as np
import tensorflow as tf

class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, max_len, ** kwargs):
        super().__init__(** kwargs)
        self.max_len    = max_len
        self.embedding_dim  = embedding_dim
        
        self.pos_encoding = self.get_positional_encoding()
        
    def get_angles(self, pos, i):
        angle_rates = 1. / tf.pow(10000, (2 * (i // 2)) / tf.cast(self.embedding_dim, tf.float32))

        return pos * angle_rates
        
    def get_positional_encoding(self):
        angle_rads = self.get_angles(
            tf.range(self.max_len, dtype = tf.float32)[:, tf.newaxis],
            tf.range(self.embedding_dim, dtype = tf.float32)[tf.newaxis, :],
        )

        angle_rads = angle_rads.numpy()
        # Applique le sinus sur les indices pairs (2 i). 
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        # Applique le sinus sur les indices pairs (2 i + 1). 
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype = tf.float32)
    
    def call(self, inputs, training = False):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]
    
    def get_config(self):
        config = super().get_config()
        config['max_len']    = self.max_len
        config['embedding_dim']  = self.embedding_dim
        return config
