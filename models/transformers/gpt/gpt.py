import tensorflow as tf
from tensorflow.keras import Model
from .decoder import Decoder
from ..components import create_padding_mask


class GPT(Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 target_vocab_size, pe_input, rate=0.1):
        super(GPT, self).__init__()

        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                               input_vocab_size, pe_input, rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size, activation='softmax')

    def call(self, inputs, training=False):
        decoder_input = inputs
        decoder_padding_mask = create_padding_mask(decoder_input)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            decoder_input, training, decoder_padding_mask)

        # (batch_size, tar_seq_len, target_vocab_size)
        final_output = self.final_layer(dec_output)

        return final_output  # , attention_weights


