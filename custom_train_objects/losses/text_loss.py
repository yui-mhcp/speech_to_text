import tensorflow as tf

class TextLoss(tf.keras.losses.Loss):
    def __init__(self, pad_value = 0, name = 'TextLoss', ** kwargs):
        kwargs['reduction'] = tf.keras.losses.Reduction.NONE
        super().__init__(name = name, ** kwargs)
        self.pad_value = pad_value
        
    def call(self, y_true, y_pred):
        skip_length = 0
        if not isinstance(y_true, (list, tuple)):
            target_length = tf.reduce_sum(tf.cast(
                tf.math.not_equal(y_true, self.pad_value), tf.int32
            ), axis = -1)
        else:
            if len(y_true) == 3: skip_length = y_true[2]
            y_true, target_length = y_true[:2]

        padding_mask    = tf.sequence_mask(
            skip_length + target_length, maxlen = tf.shape(y_pred)[1], dtype = tf.float32
        )
        if tf.reduce_any(skip_length > 0):
            padding_mask    = tf.minimum(
                padding_mask, 1 - tf.sequence_mask(
                    skip_length, maxlen = tf.shape(y_pred)[1], dtype = tf.float32
                )
            )
        
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)

        loss = loss * padding_mask

        loss = tf.reduce_sum(loss, axis = -1) / (tf.cast(target_length, tf.float32) + 1e-6)

        return loss
    
    def get_config(self):
        config = super().get_config()
        config['pad_value']     = self.pad_value
        return config
