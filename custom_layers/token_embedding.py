import tensorflow as tf

class TokenEmbedding(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, vocab_size, max_len, ** kwargs):
        super().__init__(** kwargs)
        
        self.vocab_size = vocab_size
        self.max_len    = max_len
        self.embedding_dim  = embedding_dim
        
        self.embedding_layer    = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.pos_embedding_layer    = tf.keras.layers.Embedding(max_len, embedding_dim)
        
        
    def call(self, tokens, training = False):
        embedded = self.embedding_layer(tokens, training = training)
        pos_embedded    = self.pos_embedding_layer(tf.range(0, tf.shape(tokens)[1]), training = training)
        return embedded + pos_embedded
    
    def get_config(self):
        config = super().get_config()
        config['vocab_size'] = self.vocab_size
        config['max_len']    = self.max_len
        config['embedding_dim']  = self.embedding_dim
        return config
        