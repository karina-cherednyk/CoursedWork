from tensorflow.keras import Model
import tensorflow as tf
from .encoder import Encoder
from .decoder import Decoder


class AttentionRNN(Model):
    def __init__(self,
                 input_vocab_size, target_vocab_size, embedding_dim,
                 encoder_units, decoder_units, att_units, train_len, valid_len, rate=0.1):
        super(AttentionRNN, self).__init__()
        self.encoder = Encoder(input_vocab_size, embedding_dim, encoder_units, rate)
        self.decoder = Decoder(target_vocab_size, embedding_dim, decoder_units, rate, att_units)
        self.train_len = train_len
        self.valid_len = valid_len
        self.vocab_size = target_vocab_size

    def call(self, inputs, training=False):
        if training:
          return self.call_train(inputs, training)   
        else:
          return self.call_test(inputs)

    def call_train(self, inputs, training=False):
        enc_input, dec_input = inputs
        enc_hidden = self.encoder.initialize_hidden_state()
        batch_size = tf.shape(dec_input)[0] 
        enc_output, enc_hidden = self.encoder(enc_input, enc_hidden)
        dec_hidden = enc_hidden
        predictions = tf.zeros((batch_size, 1, self.vocab_size)) 

        for t in range(self.train_len):
          tf.autograph.experimental.set_loop_options(shape_invariants=[(predictions, tf.TensorShape([None, None, self.vocab_size]))])
          predi, dec_hidden, _ = self.decoder( tf.expand_dims(dec_input[:, t], 1), dec_hidden, enc_output)
          predi = tf.expand_dims(predi, 1)
          predictions = tf.concat([ predictions, predi ], axis=1)
	
        return predictions[:,1:,:]

   
    def call_test(self, inputs):
        enc_input, dec_input = inputs
        enc_hidden = self.encoder.initialize_hidden_state()
        batch_size = tf.shape(dec_input)[0] 
        dec_input = tf.expand_dims(dec_input[:,0],1)
        enc_output, enc_hidden = self.encoder(enc_input, enc_hidden)
        dec_hidden = enc_hidden
        predictions = tf.zeros((batch_size, 1, self.vocab_size))  
     
        for i in range(self.valid_len):
          tf.autograph.experimental.set_loop_options(shape_invariants=[(predictions, tf.TensorShape([None, None, self.vocab_size]))])
          predi, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_output)
          predi = tf.expand_dims(predi, 1)
          dec_input = tf.argmax(predi, axis=-1)           
          predictions = tf.concat([ predictions, predi ], axis=1)
      
        print(dec_input.shape)
        print(predictions.shape)

        return predictions[:,1:,:]

"""
    def train_step(self, batch_data):
        loss = 0
        (enc_input, dec_input), target = batch_data
        target_len = target.shape[1]
        enc_hidden = self.encoder.initialize_hidden_state()

        with tf.GradientTape() as tape:
            enc_output, enc_hidden = self.encoder(enc_input, enc_hidden)
            dec_hidden = enc_hidden
            for t in range(target_len):
                predictions, dec_hidden, _ = self.decoder(tf.expand_dims(dec_input[:, t], 1), dec_hidden, enc_output)
                loss += self.compiled_loss(target[:, t], predictions)

        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, variables)

        self.optimizer.apply_gradients(zip(gradients, variables))
        return {m.name: m.result() for m in self.metrics}
"""