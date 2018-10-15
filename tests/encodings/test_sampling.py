import argparse

import pytest
from keras import backend as K

from config import load_parameters
from data_engine.prepare_data import build_dataset
from nmt_keras.training import train_model


def load_tests_params():
    params = load_parameters()
    params['BATCH_SIZE'] = 10
    params['WEIGHT_DECAY'] = 1e-4
    params['RECURRENT_WEIGHT_DECAY'] = 1e-4
    params['DROPOUT_P'] = 0.01
    params['RECURRENT_INPUT_DROPOUT_P'] = 0.01
    params['RECURRENT_DROPOUT_P'] = 0.01
    params['USE_NOISE'] = True
    params['NOISE_AMOUNT'] = 0.01
    params['USE_BATCH_NORMALIZATION'] = True
    params['BATCH_NORMALIZATION_MODE'] = 1
    params['SOURCE_TEXT_EMBEDDING_SIZE'] = 8
    params['TARGET_TEXT_EMBEDDING_SIZE'] = 8
    params['DECODER_HIDDEN_SIZE'] = 4
    params['ENCODER_HIDDEN_SIZE'] = 4
    params['ATTENTION_SIZE'] = params['DECODER_HIDDEN_SIZE']
    params['SKIP_VECTORS_HIDDEN_SIZE'] = params['DECODER_HIDDEN_SIZE']
    params['DOUBLE_STOCHASTIC_ATTENTION_REG'] = 0.7
    params['RELOAD'] = 0
    params['MAX_EPOCH'] = 1
    params['USE_CUDNN'] = False

    return params


def test_sampling_maxlikelihood():
    params = load_tests_params()

    params['REBUILD_DATASET'] = True
    params['INPUT_VOCABULARY_SIZE'] = 550
    params['OUTPUT_VOCABULARY_SIZE'] = 550

    params['POS_UNK'] = True
    params['HEURISTIC'] = 0
    params['ALIGN_FROM_RAW'] = True

    # Sampling params: Show some samples during training.
    params['SAMPLE_ON_SETS'] = ['train', 'val']
    params['N_SAMPLES'] = 10
    params['START_SAMPLING_ON_EPOCH'] = 0
    params['SAMPLE_EACH_UPDATES'] = 50
    params['SAMPLING'] = 'max_likelihood'

    dataset = build_dataset(params)
    params['INPUT_VOCABULARY_SIZE'] = dataset.vocabulary_len[params['INPUTS_IDS_DATASET'][0]]
    params['OUTPUT_VOCABULARY_SIZE'] = dataset.vocabulary_len[params['OUTPUTS_IDS_DATASET'][0]]
    params['MODEL_NAME'] = \
        params['TASK_NAME'] + '_' + params['SRC_LAN'] + params['TRG_LAN'] + '_' + params['MODEL_TYPE'] + \
        '_src_emb_' + str(params['SOURCE_TEXT_EMBEDDING_SIZE']) + \
        '_bidir_' + str(params['BIDIRECTIONAL_ENCODER']) + \
        '_enc_' + params['ENCODER_RNN_TYPE'] + '_*' + str(params['N_LAYERS_ENCODER']) + '_' + str(
            params['ENCODER_HIDDEN_SIZE']) + \
        '_dec_' + params['DECODER_RNN_TYPE'] + '_*' + str(params['N_LAYERS_DECODER']) + '_' + str(
            params['DECODER_HIDDEN_SIZE']) + \
        '_deepout_' + '_'.join([layer[0] for layer in params['DEEP_OUTPUT_LAYERS']]) + \
        '_trg_emb_' + str(params['TARGET_TEXT_EMBEDDING_SIZE']) + \
        '_' + params['OPTIMIZER'] + '_' + str(params['LR'])
    params['STORE_PATH'] = K.backend() + '_test_train_models/' + params['MODEL_NAME'] + '/'

    # Test several NMT-Keras utilities: train, sample, sample_ensemble, score_corpus...
    print ("Training model")
    train_model(params)
    print ("Done")


def test_sampling_multinomial():
    params = load_tests_params()

    params['REBUILD_DATASET'] = True
    params['INPUT_VOCABULARY_SIZE'] = 550
    params['OUTPUT_VOCABULARY_SIZE'] = 550

    params['POS_UNK'] = True
    params['HEURISTIC'] = 0
    params['ALIGN_FROM_RAW'] = True

    # Sampling params: Show some samples during training.
    params['SAMPLE_ON_SETS'] = ['train', 'val']
    params['N_SAMPLES'] = 10
    params['START_SAMPLING_ON_EPOCH'] = 0
    params['SAMPLE_EACH_UPDATES'] = 50
    params['SAMPLING'] = 'multinomial'

    dataset = build_dataset(params)
    params['INPUT_VOCABULARY_SIZE'] = dataset.vocabulary_len[params['INPUTS_IDS_DATASET'][0]]
    params['OUTPUT_VOCABULARY_SIZE'] = dataset.vocabulary_len[params['OUTPUTS_IDS_DATASET'][0]]
    params['MODEL_NAME'] = \
        params['TASK_NAME'] + '_' + params['SRC_LAN'] + params['TRG_LAN'] + '_' + params['MODEL_TYPE'] + \
        '_src_emb_' + str(params['SOURCE_TEXT_EMBEDDING_SIZE']) + \
        '_bidir_' + str(params['BIDIRECTIONAL_ENCODER']) + \
        '_enc_' + params['ENCODER_RNN_TYPE'] + '_*' + str(params['N_LAYERS_ENCODER']) + '_' + str(
            params['ENCODER_HIDDEN_SIZE']) + \
        '_dec_' + params['DECODER_RNN_TYPE'] + '_*' + str(params['N_LAYERS_DECODER']) + '_' + str(
            params['DECODER_HIDDEN_SIZE']) + \
        '_deepout_' + '_'.join([layer[0] for layer in params['DEEP_OUTPUT_LAYERS']]) + \
        '_trg_emb_' + str(params['TARGET_TEXT_EMBEDDING_SIZE']) + \
        '_' + params['OPTIMIZER'] + '_' + str(params['LR'])
    params['STORE_PATH'] = K.backend() + '_test_train_models/' + params['MODEL_NAME'] + '/'

    # Test several NMT-Keras utilities: train, sample, sample_ensemble, score_corpus...
    print ("Training model")
    train_model(params)
    print ("Done")


if __name__ == '__main__':
    pytest.main([__file__])
