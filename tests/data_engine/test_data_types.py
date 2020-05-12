import argparse
import os
import pytest
from tests.test_config import clean_dirs, load_transformer_test_params
from data_engine.prepare_data import build_dataset
from nmt_keras.training import train_model
from nmt_keras.apply_model import sample_ensemble, score_corpus


def test_text_features_none():
    params = load_transformer_test_params()

    # Current test params:
    params['INPUTS_TYPES_DATASET'] = ['text', 'text']
    params['OUTPUTS_TYPES_DATASET'] = ['text']

    params['REBUILD_DATASET'] = True
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

    # Test several NMT-Keras utilities: train, sample, sample_ensemble, score_corpus...
    print("Training model")
    train_model(params)
    params['RELOAD'] = 1
    print("Done")

    parser = argparse.ArgumentParser('Parser for unit testing')
    parser.dataset = os.path.join(
        params['DATASET_STORE_PATH'],
        'Dataset_' + params['DATASET_NAME'] + '_' + params['SRC_LAN'] + params['TRG_LAN'] + '.pkl')

    parser.text = os.path.join(params['DATA_ROOT_PATH'], params['TEXT_FILES']['val'] + params['SRC_LAN'])
    parser.splits = ['val']
    parser.config = params['STORE_PATH'] + '/config.pkl'
    parser.models = [params['STORE_PATH'] + '/epoch_' + str(1)]
    parser.verbose = 0
    parser.dest = None
    parser.source = os.path.join(params['DATA_ROOT_PATH'], params['TEXT_FILES']['val'] + params['SRC_LAN'])
    parser.target = os.path.join(params['DATA_ROOT_PATH'], params['TEXT_FILES']['val'] + params['TRG_LAN'])
    parser.weights = []
    parser.glossary = None

    for n_best in [True, False]:
        parser.n_best = n_best
        print("Sampling with n_best = %s " % str(n_best))
        sample_ensemble(parser, params)
        print("Done")

    print("Scoring corpus")
    score_corpus(parser, params)
    print("Done")
    clean_dirs(params)


def test_text_features_target_text():
    params = load_transformer_test_params()

    # Current test params:
    params['INPUTS_TYPES_DATASET'] = ['text', 'text']
    params['OUTPUTS_TYPES_DATASET'] = ['text-features']

    params['REBUILD_DATASET'] = True
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

    # Test several NMT-Keras utilities: train, sample, sample_ensemble, score_corpus...
    print("Training model")
    train_model(params)
    params['RELOAD'] = 1
    print("Done")

    parser = argparse.ArgumentParser('Parser for unit testing')
    parser.dataset = os.path.join(
        params['DATASET_STORE_PATH'],
        'Dataset_' + params['DATASET_NAME'] + '_' + params['SRC_LAN'] + params['TRG_LAN'] + '.pkl')

    parser.text = os.path.join(params['DATA_ROOT_PATH'], params['TEXT_FILES']['val'] + params['SRC_LAN'])
    parser.splits = ['val']
    parser.config = params['STORE_PATH'] + '/config.pkl'
    parser.models = [params['STORE_PATH'] + '/epoch_' + str(1)]
    parser.verbose = 0
    parser.dest = None
    parser.source = os.path.join(params['DATA_ROOT_PATH'], params['TEXT_FILES']['val'] + params['SRC_LAN'])
    parser.target = os.path.join(params['DATA_ROOT_PATH'], params['TEXT_FILES']['val'] + params['TRG_LAN'])
    parser.weights = []
    parser.glossary = None

    for n_best in [True, False]:
        parser.n_best = n_best
        print("Sampling with n_best = %s " % str(n_best))
        sample_ensemble(parser, params)
        print("Done")

    print("Scoring corpus")
    score_corpus(parser, params)
    print("Done")
    clean_dirs(params)


def test_text_features_state_below():
    params = load_transformer_test_params()

    # Current test params:
    params['INPUTS_TYPES_DATASET'] = ['text', 'text-features']
    params['OUTPUTS_TYPES_DATASET'] = ['text']

    params['REBUILD_DATASET'] = True
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

    # Test several NMT-Keras utilities: train, sample, sample_ensemble, score_corpus...
    print("Training model")
    train_model(params)
    params['RELOAD'] = 1
    print("Done")

    parser = argparse.ArgumentParser('Parser for unit testing')
    parser.dataset = os.path.join(
        params['DATASET_STORE_PATH'],
        'Dataset_' + params['DATASET_NAME'] + '_' + params['SRC_LAN'] + params['TRG_LAN'] + '.pkl')

    parser.text = os.path.join(params['DATA_ROOT_PATH'], params['TEXT_FILES']['val'] + params['SRC_LAN'])
    parser.splits = ['val']
    parser.config = params['STORE_PATH'] + '/config.pkl'
    parser.models = [params['STORE_PATH'] + '/epoch_' + str(1)]
    parser.verbose = 0
    parser.dest = None
    parser.source = os.path.join(params['DATA_ROOT_PATH'], params['TEXT_FILES']['val'] + params['SRC_LAN'])
    parser.target = os.path.join(params['DATA_ROOT_PATH'], params['TEXT_FILES']['val'] + params['TRG_LAN'])
    parser.weights = []
    parser.glossary = None

    for n_best in [True, False]:
        parser.n_best = n_best
        print("Sampling with n_best = %s " % str(n_best))
        sample_ensemble(parser, params)
        print("Done")

    print("Scoring corpus")
    score_corpus(parser, params)
    print("Done")
    clean_dirs(params)


def test_text_features_src():
    params = load_transformer_test_params()

    # Current test params:
    params['INPUTS_TYPES_DATASET'] = ['text-features', 'text']
    params['OUTPUTS_TYPES_DATASET'] = ['text']

    params['REBUILD_DATASET'] = True
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

    # Test several NMT-Keras utilities: train, sample, sample_ensemble, score_corpus...
    print("Training model")
    train_model(params)
    params['RELOAD'] = 1
    print("Done")

    parser = argparse.ArgumentParser('Parser for unit testing')
    parser.dataset = os.path.join(
        params['DATASET_STORE_PATH'],
        'Dataset_' + params['DATASET_NAME'] + '_' + params['SRC_LAN'] + params['TRG_LAN'] + '.pkl')

    parser.text = os.path.join(params['DATA_ROOT_PATH'], params['TEXT_FILES']['val'] + params['SRC_LAN'])
    parser.splits = ['val']
    parser.config = params['STORE_PATH'] + '/config.pkl'
    parser.models = [params['STORE_PATH'] + '/epoch_' + str(1)]
    parser.verbose = 0
    parser.dest = None
    parser.source = os.path.join(params['DATA_ROOT_PATH'], params['TEXT_FILES']['val'] + params['SRC_LAN'])
    parser.target = os.path.join(params['DATA_ROOT_PATH'], params['TEXT_FILES']['val'] + params['TRG_LAN'])
    parser.weights = []
    parser.glossary = None

    for n_best in [True, False]:
        parser.n_best = n_best
        print("Sampling with n_best = %s " % str(n_best))
        sample_ensemble(parser, params)
        print("Done")

    print("Scoring corpus")
    score_corpus(parser, params)
    print("Done")
    clean_dirs(params)


if __name__ == '__main__':
    pytest.main([__file__])
