import tensorflow as tf
from tensorflow.keras.layers import Layer

class AttentionLayer(Layer):
    """
    Bahdanau-style Attention Layer
    """
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='att_weight', 
                                 shape=(input_shape[-1], input_shape[-1]),
                                 initializer='glorot_uniform', trainable=True)
        self.b = self.add_weight(name='att_bias', 
                                 shape=(input_shape[-1],),
                                 initializer='zeros', trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        # x: (batch_size, time_steps, features)
        e = tf.nn.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        a = tf.nn.softmax(e, axis=1)
        output = a * x
        return tf.reduce_sum(output, axis=1), a  # (context_vector, attention_weights)

class FeatureAttentionLayer(Layer):
    """
    Feature-wise Attention Layer
    """
    def __init__(self, **kwargs):
        super(FeatureAttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W_f = self.add_weight(name="W_f", 
                                  shape=(input_shape[-1], 1),
                                  initializer="uniform", trainable=True)
        super(FeatureAttentionLayer, self).build(input_shape)

    def call(self, x):
        feature_weights = tf.nn.softmax(tf.matmul(x, self.W_f), axis=1)
        output = x * feature_weights
        return output