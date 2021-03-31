import re

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from .tokenizer import CustomTokenizer
import params
from .preprocess import preprocess
import tensorflow as tf
from .generator import DataGenerator


def get_index(line, max_input_len):
    size = len(line.split())
    for i in range(5, max_input_len+1, 5):
        if size <= i:
            return i
    return -1


def _get_keys(args):
    if args:
        args[params.encoder_input_valid] = []
        args[params.decoder_input_valid] = []
        args[params.decoder_output_valid] = []
        args[params.num_samples_varlen_valid] = {}
        args[params.num_samples_valid] = 0

        return args, params.valid_generator, \
               params.encoder_input_valid, params.decoder_input_valid, \
               params.decoder_output_valid, params.num_samples_valid, \
               params.num_samples_varlen_valid
    else:
        args = {
            params.encoder_input: [],
            params.decoder_input: [],
            params.decoder_output: [],
            params.num_samples_varlen: {},
            params.num_samples: 0,
            params.input_tokenizer: CustomTokenizer(params.to_exclude),
            params.target_tokenizer: CustomTokenizer(params.to_exclude)}

        return args, params.train_generator, \
               params.encoder_input, params.decoder_input, \
               params.decoder_output, params.num_samples, \
               params.num_samples_varlen


def make_batches(text_pairs, max_input_len=params.max_input_len,args=None):
    by_len = {}
    is_train = not bool(args)

    args, dg, einp, dinp, dout, num, num_varlen = _get_keys(args)
    input_tokenizer = args[params.input_tokenizer]
    target_tokenizer = args[params.target_tokenizer]

    for i in range(5, max_input_len+1, 5):
        by_len[i] = [[], []]

    max_target_len = 0

    for input_text, target_text in text_pairs:
        input_text, target_text = preprocess(input_text), preprocess(target_text)
        if re.sub(r'[0-9\.?!]', '', input_text).strip() == '':
            continue
        target_text = params.start_token + ' ' + target_text + ' ' + params.end_token
        i = get_index(input_text, max_input_len)
        if i == -1:
            continue

        by_len[i][0].append(input_text)
        by_len[i][1].append(target_text)
        max_target_len = max(max_target_len, len(target_text.split(' ')))

    for i in range(5, max_input_len+1, 5):
        input_texts, target_texts = by_len[i]
        size = len(input_texts)
        if size == 0:
            continue

        if is_train:
            input_tokenizer.fit_on_texts(input_texts)
            target_tokenizer.fit_on_texts(target_texts)

        input_sequences = input_tokenizer.texts_to_sequences(input_texts)
        target_sequences = target_tokenizer.texts_to_sequences(target_texts)

        encoder_input_sequences = pad_sequences(input_sequences, padding='post').astype(np.int64)
        decoder_input_sequences = pad_sequences(target_sequences, padding='post', maxlen=max_target_len).astype(
            np.int64)

        decoder_output_sequences = np.concatenate((decoder_input_sequences[:, 1:], np.zeros((size, 1))), axis=1)

        s = np.arange(size - size % params.batch_size).reshape(-1, params.batch_size)
        if size % params.batch_size != 0:
            s = s.tolist() + np.arange(s.shape[0] * s.shape[1], size)[np.newaxis, :].tolist()

        for indices in s:
            args[einp].append(encoder_input_sequences[indices, :])
            args[dinp].append(decoder_input_sequences[indices, :])
            args[dout].append(decoder_output_sequences[indices, :])

        args[num] += size
        args[num_varlen] = size

    args[dg] = DataGenerator(args, is_train)
    if is_train:
        args[params.input_vocab_size] = len(input_tokenizer.word_index) + 1
        args[params.target_vocab_size] = len(target_tokenizer.word_index) + 1

    return args
