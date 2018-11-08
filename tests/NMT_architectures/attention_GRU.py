import argparse

import pytest
from keras import backend as K

from config import load_parameters
from data_engine.prepare_data import build_dataset
from nmt_keras.training import train_model
from nmt_keras.apply_model import sample_ensemble, score_corpus


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


def test_GRU_add():
    params = load_tests_params()

    # Current test params: Single layered LSTM - ConditionalGRU
    params['BIDIRECTIONAL_ENCODER'] = True
    params['N_LAYERS_ENCODER'] = 1
    params['BIDIRECTIONAL_DEEP_ENCODER'] = True
    params['ENCODER_RNN_TYPE'] = 'GRU'
    params['DECODER_RNN_TYPE'] = 'GRU'
    params['N_LAYERS_DECODER'] = 1
    params['ATTENTION_MODE'] = 'add'

    params['REBUILD_DATASET'] = True
    dataset = build_dataset(params)
    params['INPUT_VOCABULARY_SIZE'] = dataset.vocabulary_len[params['INPUTS_IDS_DATASET'][0]]
    params['OUTPUT_VOCABULARY_SIZE'] = dataset.vocabulary_len[params['OUTPUTS_IDS_DATASET'][0]]
    params['MODEL_NAME'] = \
        params['TASK_NAME'] + '_' + params['SRC_LAN'] + params['TRG_LAN'] + '_' + params['MODEL_TYPE'] + \
        '_src_emb_' + str(params['SOURCE_TEXT_EMBEDDING_SIZE']) + \
        '_bidir_' + str(params['BIDIRECTIONAL_ENCODER']) + \
        '_enc_' + params['ENCODER_RNN_TYPE'] + '_*' + str(params['N_LAYERS_ENCODER']) + '_' + str(params['ENCODER_HIDDEN_SIZE']) + \
        '_dec_' + params['DECODER_RNN_TYPE'] + '_*' + str(params['N_LAYERS_DECODER']) + '_' + str(params['DECODER_HIDDEN_SIZE']) + params['ATTENTION_MODE'] + \
        '_deepout_' + '_'.join([layer[0] for layer in params['DEEP_OUTPUT_LAYERS']]) + \
        '_trg_emb_' + str(params['TARGET_TEXT_EMBEDDING_SIZE']) + \
        '_' + params['OPTIMIZER'] + '_' + str(params['LR'])
    params['STORE_PATH'] = K.backend() + '_test_train_models/' + params['MODEL_NAME'] + '/'

    # Test several NMT-Keras utilities: train, sample, sample_ensemble, score_corpus...
    print ("Training model")
    train_model(params)
    params['RELOAD'] = 1
    print ("Done")

    parser = argparse.ArgumentParser('Parser for unit testing')
    parser.dataset = params['DATASET_STORE_PATH'] + '/Dataset_' + params['DATASET_NAME'] + '_' + params['SRC_LAN'] + params['TRG_LAN'] + '.pkl'

    parser.text = params['DATA_ROOT_PATH'] + '/' + params['TEXT_FILES']['val'] + params['SRC_LAN']
    parser.splits = ['val']
    parser.config = params['STORE_PATH'] + '/config.pkl'
    parser.models = [params['STORE_PATH'] + '/epoch_' + str(1)]
    parser.verbose = 0
    parser.dest = None
    parser.source = params['DATA_ROOT_PATH'] + '/' + params['TEXT_FILES']['val'] + params['SRC_LAN']
    parser.target = params['DATA_ROOT_PATH'] + '/' + params['TEXT_FILES']['val'] + params['TRG_LAN']
    parser.weights = []
    parser.glossary = None

    for n_best in [True, False]:
        parser.n_best = n_best
        print ("Sampling with n_best = %s " % str(n_best))
        sample_ensemble(parser, params)
        print ("Done")

    print ("Scoring corpus")
    score_corpus(parser, params)
    print ("Done")


def test_ConditionalGRU_dot():
    params = load_tests_params()

    # Current test params: Single layered LSTM - ConditionalGRU
    params['BIDIRECTIONAL_ENCODER'] = True
    params['N_LAYERS_ENCODER'] = 1
    params['BIDIRECTIONAL_DEEP_ENCODER'] = True
    params['ENCODER_RNN_TYPE'] = 'GRU'
    params['DECODER_RNN_TYPE'] = 'GRU'
    params['N_LAYERS_DECODER'] = 1
    params['ATTENTION_MODE'] = 'dot'

    params['REBUILD_DATASET'] = True
    dataset = build_dataset(params)
    params['INPUT_VOCABULARY_SIZE'] = dataset.vocabulary_len[params['INPUTS_IDS_DATASET'][0]]
    params['OUTPUT_VOCABULARY_SIZE'] = dataset.vocabulary_len[params['OUTPUTS_IDS_DATASET'][0]]
    params['MODEL_NAME'] = \
        params['TASK_NAME'] + '_' + params['SRC_LAN'] + params['TRG_LAN'] + '_' + params['MODEL_TYPE'] + \
        '_src_emb_' + str(params['SOURCE_TEXT_EMBEDDING_SIZE']) + \
        '_bidir_' + str(params['BIDIRECTIONAL_ENCODER']) + \
        '_enc_' + params['ENCODER_RNN_TYPE'] + '_*' + str(params['N_LAYERS_ENCODER']) + '_' + str(params['ENCODER_HIDDEN_SIZE']) + \
        '_dec_' + params['DECODER_RNN_TYPE'] + '_*' + str(params['N_LAYERS_DECODER']) + '_' + str(params['DECODER_HIDDEN_SIZE']) + params['ATTENTION_MODE'] + \
        '_deepout_' + '_'.join([layer[0] for layer in params['DEEP_OUTPUT_LAYERS']]) + \
        '_trg_emb_' + str(params['TARGET_TEXT_EMBEDDING_SIZE']) + \
        '_' + params['OPTIMIZER'] + '_' + str(params['LR'])
    params['STORE_PATH'] = K.backend() + '_test_train_models/' + params['MODEL_NAME'] + '/'

    # Test several NMT-Keras utilities: train, sample, sample_ensemble, score_corpus...
    print ("Training model")
    train_model(params)
    params['RELOAD'] = 1
    print ("Done")

    parser = argparse.ArgumentParser('Parser for unit testing')
    parser.dataset = params['DATASET_STORE_PATH'] + '/Dataset_' + params['DATASET_NAME'] + '_' + params['SRC_LAN'] + params['TRG_LAN'] + '.pkl'

    parser.text = params['DATA_ROOT_PATH'] + '/' + params['TEXT_FILES']['val'] + params['SRC_LAN']
    parser.splits = ['val']
    parser.config = params['STORE_PATH'] + '/config.pkl'
    parser.models = [params['STORE_PATH'] + '/epoch_' + str(1)]
    parser.verbose = 0
    parser.dest = None
    parser.source = params['DATA_ROOT_PATH'] + '/' + params['TEXT_FILES']['val'] + params['SRC_LAN']
    parser.target = params['DATA_ROOT_PATH'] + '/' + params['TEXT_FILES']['val'] + params['TRG_LAN']
    parser.weights = []
    parser.glossary = None

    for n_best in [True, False]:
        parser.n_best = n_best
        print ("Sampling with n_best = %s " % str(n_best))
        sample_ensemble(parser, params)
        print ("Done")

    print ("Scoring corpus")
    score_corpus(parser, params)
    print ("Done")


def test_ConditionalGRU_scaled():
    params = load_tests_params()

    # Current test params: Single layered LSTM - ConditionalGRU
    params['BIDIRECTIONAL_ENCODER'] = True
    params['N_LAYERS_ENCODER'] = 1
    params['BIDIRECTIONAL_DEEP_ENCODER'] = True
    params['ENCODER_RNN_TYPE'] = 'GRU'
    params['DECODER_RNN_TYPE'] = 'GRU'
    params['N_LAYERS_DECODER'] = 1
    params['ATTENTION_MODE'] = 'scaled-dot'

    params['REBUILD_DATASET'] = True
    dataset = build_dataset(params)
    params['INPUT_VOCABULARY_SIZE'] = dataset.vocabulary_len[params['INPUTS_IDS_DATASET'][0]]
    params['OUTPUT_VOCABULARY_SIZE'] = dataset.vocabulary_len[params['OUTPUTS_IDS_DATASET'][0]]
    params['MODEL_NAME'] = \
        params['TASK_NAME'] + '_' + params['SRC_LAN'] + params['TRG_LAN'] + '_' + params['MODEL_TYPE'] + \
        '_src_emb_' + str(params['SOURCE_TEXT_EMBEDDING_SIZE']) + \
        '_bidir_' + str(params['BIDIRECTIONAL_ENCODER']) + \
        '_enc_' + params['ENCODER_RNN_TYPE'] + '_*' + str(params['N_LAYERS_ENCODER']) + '_' + str(params['ENCODER_HIDDEN_SIZE']) + \
        '_dec_' + params['DECODER_RNN_TYPE'] + '_*' + str(params['N_LAYERS_DECODER']) + '_' + str(params['DECODER_HIDDEN_SIZE']) + params['ATTENTION_MODE'] + \
        '_deepout_' + '_'.join([layer[0] for layer in params['DEEP_OUTPUT_LAYERS']]) + \
        '_trg_emb_' + str(params['TARGET_TEXT_EMBEDDING_SIZE']) + \
        '_' + params['OPTIMIZER'] + '_' + str(params['LR'])
    params['STORE_PATH'] = K.backend() + '_test_train_models/' + params['MODEL_NAME'] + '/'

    # Test several NMT-Keras utilities: train, sample, sample_ensemble, score_corpus...
    print ("Training model")
    train_model(params)
    params['RELOAD'] = 1
    print ("Done")

    parser = argparse.ArgumentParser('Parser for unit testing')
    parser.dataset = params['DATASET_STORE_PATH'] + '/Dataset_' + params['DATASET_NAME'] + '_' + params['SRC_LAN'] + params['TRG_LAN'] + '.pkl'

    parser.text = params['DATA_ROOT_PATH'] + '/' + params['TEXT_FILES']['val'] + params['SRC_LAN']
    parser.splits = ['val']
    parser.config = params['STORE_PATH'] + '/config.pkl'
    parser.models = [params['STORE_PATH'] + '/epoch_' + str(1)]
    parser.verbose = 0
    parser.dest = None
    parser.source = params['DATA_ROOT_PATH'] + '/' + params['TEXT_FILES']['val'] + params['SRC_LAN']
    parser.target = params['DATA_ROOT_PATH'] + '/' + params['TEXT_FILES']['val'] + params['TRG_LAN']
    parser.weights = []
    parser.glossary = None

    for n_best in [True, False]:
        parser.n_best = n_best
        print ("Sampling with n_best = %s " % str(n_best))
        sample_ensemble(parser, params)
        print ("Done")

    print ("Scoring corpus")
    score_corpus(parser, params)
    print ("Done")


if __name__ == '__main__':
    pytest.main([__file__])
