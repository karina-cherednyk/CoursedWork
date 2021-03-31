import params
from .preprocess import preprocess_filter, seq_to_ds
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.sequence import pad_sequences


def make_batches(text_pairs, max_input_len=params.max_input_len, args=None, const_size=False):
    is_train = not bool(args)

    input_texts, target_texts = preprocess_filter(text_pairs, max_input_len)

    if is_train:
        args = {}
        input_tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            input_texts, target_vocab_size=2 ** 13)

        target_tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            target_texts, reserved_tokens=[params.start_token, params.end_token],
            target_vocab_size=2 ** 13)

        args[params.input_tokenizer] = input_tokenizer
        args[params.target_tokenizer] = target_tokenizer
    else:
        input_tokenizer = args[params.input_tokenizer]
        target_tokenizer = args[params.target_tokenizer]

    input_seq = list(map(input_tokenizer.encode, input_texts))
    target_seq = list(map(target_tokenizer.encode, target_texts))
    dataset, target_len = seq_to_ds(input_seq, target_seq, const_size)
    

    if is_train:
        args[params.train_generator] = dataset
        args[params.input_vocab_size] = input_tokenizer.vocab_size + 1
        args[params.target_vocab_size] = target_tokenizer.vocab_size + 1
        args[params.train_seq_len] = target_len
    else:
        args[params.valid_generator] = dataset
        args[params.valid_seq_len] = target_len
    return args
