import main
import time
from metrics import loss_function, accuracy_function, bleu_score
from models import AttentionRNN
import tensorflow as tf
from dataset.tf_datasets import getEnRoLists
from vectorization import var_batches, ds_batches
import params
import os
import json

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

learning_rate = CustomSchedule(256)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, 
                                     epsilon=1e-9)


in_texts, tar_texts = getEnRoLists()
in_valid, tar_valid = getEnRoLists(False)

encoder_units = 640
decoder_units = 640
embedding_dim = 256  # context size
att_units = 10
dropout_rate = 0.1


weights_dir = 'weights/ds/attention'
history_dir = 'history/ds/attention'

os.makedirs(weights_dir, exist_ok=True) 
os.makedirs(history_dir, exist_ok=True)

import sys
if len(sys.argv) > 1:
  maxlen = int(sys.argv[1])
else:
  maxlen = 16


args = ds_batches(zip(in_texts, tar_texts), maxlen)
args = ds_batches(zip(in_valid, tar_valid), maxlen, args)
dg = args[params.train_generator]
vdg = args[params.valid_generator]
input_vocab_size = args[params.input_vocab_size]
target_vocab_size = args[params.target_vocab_size]
train_len = args[params.train_seq_len]
valid_len = args[params.valid_seq_len]


rnn = AttentionRNN(input_vocab_size, target_vocab_size, embedding_dim,
                   encoder_units, decoder_units, att_units, train_len, valid_len, dropout_rate)

rnn.compile(optimizer=optimizer, loss=loss_function, metrics=[accuracy_function, bleu_score])
history = rnn.fit(dg, epochs=params.epochs, validation_data = vdg)

rnn.save_weights(weights_dir+'/w'+str(maxlen))
json.dump(history.history, open(history_dir+'/h'+str(maxlen), 'w'))
