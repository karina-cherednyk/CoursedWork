import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import layers
from .attention_layer import BahdanauAttention


class Decoder(Layer):
    def __init__(self, vocab_size, embedding_dim, dec_units,  rate=0.1, att_units=10):
        super(Decoder, self).__init__()
        self.dec_units = dec_units
        self.d_model = embedding_dim
        self.embedding = layers.Embedding(vocab_size, self.d_model, mask_zero=True)
        self.gru = layers.GRU(self.dec_units,
                              return_sequences=True,
                              return_state=True,
                              dropout=rate,
                              recurrent_initializer='glorot_uniform')

        self.fc = layers.Dense(vocab_size, activation='softmax')

        # used for attention
        self.attention = BahdanauAttention(att_units)
        self.dropout1 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, hidden, enc_output, training):
        context_vector, attention_weights = self.attention(hidden, enc_output)
        context_vector = self.dropout1(context_vector, training=training)

        x = self.embedding(inputs)
        mask = self.embedding.compute_mask(inputs)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        x = tf.concat([context_vector, x], axis=-1)

        output, state = self.gru(x, mask=mask)
        output = tf.reshape(output, (-1, output.shape[2]))

        x = self.fc(output)

        return x, state, attention_weights
