import params
from tensorflow.keras.models import load_model


def save_model(args, dir_name):
    args['model'].save('saved_models/' + dir_name + '/model')
    args[params.encoder_model].save('saved_models/' + dir_name + '/encoder_model')
    args[params.decoder_model].save('saved_models/' + dir_name + '/decoder_model')


def load_model(args, dir_name):
    args['model'] = load_model('saved_models/' + dir_name + '/model')
    args[params.encoder_model] = load_model('saved_models/' + dir_name + '/encoder_model')
    args[params.decoder_model] = load_model('saved_models/' + dir_name + '/decoder_model')
