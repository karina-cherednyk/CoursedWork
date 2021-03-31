import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import layers
import params


class Encoder(Layer):
    def __init__(self, vocab_size, embedding_dim, enc_units,rate):
        super(Encoder, self).__init__()
        self.enc_units = enc_units
        self.d_model = embedding_dim
        self.embedding = layers.Embedding(vocab_size, self.d_model, mask_zero=True)
        self.gru = layers.GRU(self.enc_units,
                              return_sequences=True,
                              return_state=True,
                              dropout=rate,
                              recurrent_initializer='glorot_uniform')

    def call(self, inputs, hidden):
        x = self.embedding(inputs)
        mask = self.embedding.compute_mask(inputs)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        output, state = self.gru(x, initial_state=hidden, mask=mask)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((params.batch_size, self.enc_units), tf.float32)
