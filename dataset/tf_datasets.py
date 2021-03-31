import tensorflow_datasets as tfds
import params


def getEnRoData():
    dataset = tfds.load(name='wmt16_translate/ro-en',
                        data_dir='../datasets',
                        download=False,
                        as_supervised=True)
    return dataset['train'], dataset['test']


def getEnRoLists(train=True):
    if train:
      train_dataset, _ = getEnRoData()
      size = params.train_size
    else:
      _, train_dataset = getEnRoData()
      size = params.valid_size

    in_texts = []
    out_texts = []
    for inp, tar in train_dataset.take(size):
        in_texts.append(inp.numpy().decode('utf-8'))
        out_texts.append(tar.numpy().decode('utf-8'))

    return in_texts, out_texts
