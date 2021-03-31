import numpy as np
import tensorflow as tf

batches = 5000
batch_size = 64

train_size = batches * batch_size
valid_size = (batches//50) * batch_size

# possible args keys
input_vocab_size = 'Number of unique input lang tokens'
target_vocab_size = 'Number of unique targets lang tokens'
max_encoder_seq_length = 'Max sequence length for inputs'
max_decoder_seq_length = 'Max sequence length for outputs'
input_tokenizer = 'Input tokens index'
target_tokenizer = 'Target tokens index'

train_generator = 'Train data generator'
valid_generator = 'Validation data generator'

encoder_input = 'Encoder input data'
decoder_input = 'Decoder input data'
decoder_output = 'Decoder/Target output data'

encoder_input_valid = 'Encoder input validation'
decoder_input_valid = 'Decoder input validation'
decoder_output_valid = 'Decoder output validation'

num_samples = 'Number of samples'
num_samples_valid = 'Num of validation samples'

num_samples_varlen = 'Number of samples of length'
num_samples_varlen_valid = 'Number of validation samples of length'

encoder_model = 'Encoder model'
decoder_model = 'Decoder model'
one_state = 'Has one hidden state'

max_input_len = 16
train_seq_len = 'Length of padded train target sequences'
valid_seq_len = 'Length of padded validation target sequence' 

# possible model params
latent_dim = 640


# vectorization params
to_exclude = '"#$%&()*+-/:;=@[\\]^_`{|}~\t\n'
to_tokenize = '.,:;!?<>'
start_token = '<SOT>'
end_token = '<EOT>'

epochs = 32

def print_args(args):
    for key in args:
        if isinstance(args[key], (np.ndarray, tf.Tensor)):
            print(key, ': ', args[key].shape)
        elif isinstance(args[key], list):
            print(key, ': ', len(args[key]))
        elif str(type(args[key])) == "<class 'Tokenizer'>":
            print(key, ': ', args[key].word_index)
        else:
            print(key, ': ', args[key])
