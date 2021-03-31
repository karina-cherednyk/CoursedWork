import re

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import params
import tensorflow as tf
from .preprocess import preprocess_filter, seq_to_ds


def _get_keys(args):
    if args:
        args[params.num_samples_valid] = 0

        return args, params.valid_generator, params.num_samples_valid, params.valid_seq_len
    else:
        args = {
            params.num_samples: 0,
            params.input_tokenizer: Tokenizer(filters=params.to_exclude),
            params.target_tokenizer: Tokenizer(filters=params.to_exclude)
        }

        return args, params.train_generator, params.num_samples, params.train_seq_len


def make_batches(text_pairs, max_input_len=params.max_input_len, args=None, const_size=False):
    is_train = not bool(args)
    args, ds, num, target_seq_len = _get_keys(args)
    input_tokenizer = args[params.input_tokenizer]
    target_tokenizer = args[params.target_tokenizer]

    input_texts, target_texts = preprocess_filter(text_pairs, max_input_len)

    args[num] = input_texts.shape[0]

    if is_train:
        input_tokenizer.fit_on_texts(input_texts)
        target_tokenizer.fit_on_texts(target_texts)

    input_seq = input_tokenizer.texts_to_sequences(input_texts)
    target_seq = target_tokenizer.texts_to_sequences(target_texts)
    
    args[ds], target_len = seq_to_ds(input_seq, target_seq, const_size)
    args[target_seq_len] = target_len
    if is_train:
        args[params.input_vocab_size] = len(input_tokenizer.word_index) + 1
        args[params.target_vocab_size] = len(target_tokenizer.word_index) + 1
    return args
