import tensorflow as tf
import numpy as np
from tensorflow.keras import Model

from ..components import create_masks
from .decoder import Decoder
from .encoder import Encoder


class Transformer(Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 target_vocab_size, pe_input, pe_target, target_len, rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                               input_vocab_size, pe_input, rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                               target_vocab_size, pe_target, rate)
        self.vocab_size = target_vocab_size
        self.target_len = target_len

        self.final_layer = tf.keras.layers.Dense(target_vocab_size, activation='softmax')

    def call(self, inputs, training=False):
        if training:
          return self.call_train(inputs, training)   
        else:
          return self.call_test(inputs)

    def call_train(self, inputs, training=False):
        encoder_input, decoder_input = inputs
        enc_padding_mask, look_ahead_mask, dec_padding_mask = create_masks(encoder_input, decoder_input)

        encoder_output = self.encoder(encoder_input, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

        dec_output, attention_weights = self.decoder(
            decoder_input, encoder_output, training, look_ahead_mask, dec_padding_mask)

        # (batch_size, tar_seq_len, target_vocab_size)
        final_output = self.final_layer(dec_output)

        return final_output 

   
    def call_test(self, inputs):
       encoder_input, decoder_input = inputs
       batch_size = tf.shape(decoder_input)[0] 
       decoder_input = tf.expand_dims(decoder_input[:,0],1)
       predictions = tf.zeros((batch_size, 1, self.vocab_size))  
     
       for i in range(self.target_len):
         tf.autograph.experimental.set_loop_options(shape_invariants=[(predictions, tf.TensorShape([None, None, self.vocab_size]))])
         predi = self.call_train( (encoder_input, decoder_input), False )
         predi = predi[:,-1:,:]
         decoder_input = tf.concat([decoder_input, tf.argmax(predi,-1) ], axis=-1 )
         
         predictions = tf.concat([ predictions, predi ], axis=1)
      
       return predictions[:,1:,:]

""" 
    def train_step(self, data):
        input, target = data
        targ_real = target[:, 1:]
        targ_input = target[:,:-1]

        with tf.GradientTape() as tape:
            predictions = self((input,targ_input), training=True)  
            loss = self.compiled_loss(targ_real, predictions)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(targ_real, predictions)
        
        return {m.name: m.result() for m in self.metrics}
"""