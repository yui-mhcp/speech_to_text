import tensorflow as tf

class TextLoss(tf.keras.losses.Loss):
    def __init__(self, pad_value = 0, name = 'TextLoss', **kwargs):
        super().__init__(name = name, ** kwargs)
        self.pad_value = pad_value
        
    def call(self, y_true, y_pred):
        if not isinstance(y_true, (list, tuple)):
            target_length = tf.reduce_sum(tf.cast(tf.math.not_equal(y_true, self.pad_value), tf.int32), axis = -1)
        else:
            y_true, target_length = y_true

        padding_mask    = tf.sequence_mask(
            target_length, maxlen = tf.reduce_max(target_length), dtype = tf.float32
        )
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        
        loss = loss * padding_mask
        loss = tf.reduce_sum(loss, axis = -1) / (tf.cast(target_length, tf.float32) + 1e-6)

        return tf.reduce_mean(loss)
    
    def get_config(self):
        config = super().get_config()
        config['pad_value']     = self.pad_value
        return config
