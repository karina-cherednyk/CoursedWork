import re
import numpy as np
import params
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences


def preprocess(text):
    return re.sub(r'([' + params.to_tokenize + '])', r' \1 ', text)[:params.max_input_len]


def prepare(text):
    return params.start_token + ' ' + preprocess(text) + ' ' + params.end_token


def get_size(line):
    return len(line.split())


def split_two(inp, tar):
    inp2 = tar[:-1]
    tar2 = tar[1:]
    return (inp, inp2), tar2


def preprocess_filter(text_pairs, max_input_len):
    input_texts, target_texts = zip(*list(text_pairs))

    # sizes = np.array(list(map(get_size, input_texts)))
    # indices = np.where(sizes <= max_input_len)

    input_texts = list(map(preprocess, input_texts))
    target_texts = list(map(prepare, target_texts))
    return input_texts, target_texts


def seq_to_ds(input_sequences, target_sequences, const_size):
    buffer_size = 10000

    input_sequences = pad_sequences(input_sequences, padding='post')
    target_sequences = pad_sequences(target_sequences, padding='post')

    if const_size:
        size = max(input_sequences.shape[1], target_sequences.shape[1])
        input_sequences = pad_sequences(input_sequences, padding='post', maxlen=size)
        target_sequences = pad_sequences(target_sequences, padding='post', maxlen=size)


    dataset_input = tf.data.Dataset.from_tensor_slices(tf.cast(input_sequences, tf.int64))
    dataset_target = tf.data.Dataset.from_tensor_slices(tf.cast(target_sequences, tf.int64))
    dataset = tf.data.Dataset.zip((dataset_input, dataset_target))

    if not const_size:
        dataset = dataset.map(split_two)
    dataset = dataset.cache()
    dataset = dataset.shuffle(buffer_size).padded_batch(params.batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset, target_sequences.shape[1] - 1
