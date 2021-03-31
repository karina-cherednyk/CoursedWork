import re

import params


class CustomTokenizer:
    def __init__(self, filters=params.to_exclude):
        self.word_index = {}
        self.index_word = {}
        self.counter = 1
        self.filters = '[' + filters + ']'

    def fit_on_texts(self, texts):
        for text in texts:
            text = re.sub(self.filters, '', text)
            for word in text.split():
                if word not in self.word_index:
                    self.word_index[word] = self.counter
                    self.index_word[self.counter] = word
                    self.counter += 1

    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            seq = []
            for word in text.split():
                if word in self.word_index:
                    seq.append(self.word_index[word])
            sequences.append(seq)
        return sequences

    def sequences_to_texts(self, sequences):
        texts = []
        for seq in sequences:
            sent = []
            for index in seq:
                if index in self.index_word:
                    sent.append(self.index_word[index])
            texts.append(' '.join(sent))
        return texts
