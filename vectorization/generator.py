import numpy as np
from tensorflow import keras
import params


class DataGenerator(keras.utils.Sequence):

    def __init__(self, args, train=True):
        if train:
            self.input1 = args[params.encoder_input]
            self.input2 = args[params.decoder_input]
            self.output = args[params.decoder_output]
        else:
            self.input1 = args[params.encoder_input_valid]
            self.input2 = args[params.decoder_input_valid]
            self.output = args[params.decoder_output_valid]

        self.num_batches = len(self.input1)
        self.indexes = np.arange(self.num_batches)

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return self.num_batches

    def __getitem__(self, index):
        """Generate one batch of data"""
        return (self.input1[index], self.input2[index]), self.output[index]

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        np.random.shuffle(self.indexes)
