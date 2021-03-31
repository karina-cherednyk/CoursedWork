import main
import tensorflow_datasets as tfds
import tensorflow as tf
import time
from models.transformers.components import create_masks
from metrics import loss_function, accuracy_function, bleu_score
import numpy as np
from models import GPT
from dataset.tf_datasets import getEnRoLists
from vectorization import var_batches, ds_batches, subword_batches
import params
import json

in_texts, tar_texts = getEnRoLists()
in_valid, tar_valid = getEnRoLists(False)


num_layers = 12  # num of encoder/decoder layers
d_model = 768  # context size
dff = 3072  # position-wise feedforward network hidden dims
num_heads = 12  # number of attention heads
dropout_rate = 0.1


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

learning_rate = CustomSchedule(d_model)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, 
                                     epsilon=1e-9)

weights_dir = 'weights/subwords/gpt'
history_dir = 'history/subwords/gpt'

import os
os.makedirs(weights_dir, exist_ok=True) 
os.makedirs(history_dir, exist_ok=True)

import sys
if len(sys.argv) > 1:
  maxlen = int(sys.argv[1])
else:
  maxlen = 16

args = subword_batches(zip(in_texts, tar_texts), maxlen, const_size=True)
args = subword_batches(zip(in_valid, tar_valid), maxlen, args, const_size=True)
dg = args[params.train_generator]
vdg = args[params.valid_generator]
input_vocab_size = args[params.input_vocab_size]
target_vocab_size = args[params.target_vocab_size]



gpt = GPT(num_layers, d_model, num_heads, dff,
          input_vocab_size, target_vocab_size,
          pe_input=input_vocab_size,
          rate=dropout_rate)

gpt.compile(optimizer=optimizer, loss=loss_function, metrics=[accuracy_function, bleu_score])
history = gpt.fit(dg, epochs=params.epochs, validation_data=vdg)

gpt.save_weights(weights_dir+'/w'+str(maxlen))
json.dump(history.history, open(history_dir+'/h'+str(maxlen), 'w'))
