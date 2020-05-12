import argparse
import os
import pytest
from tests.test_config import load_tests_params, clean_dirs
from data_engine.prepare_data import build_dataset
from nmt_keras.training import train_model
from nmt_keras.apply_model import sample_ensemble, score_corpus


def test_transformer():
    params = load_tests_params()

    # Current test params: Transformer
    params['MODEL_TYPE'] = 'Transformer'
    params['N_LAYERS_ENCODER'] = 2
    params['N_LAYERS_DECODER'] = 2
    params['MULTIHEAD_ATTENTION_ACTIVATION'] = 'relu'
    params['MODEL_SIZE'] = 8
    params['FF_SIZE'] = params['MODEL_SIZE'] * 4
    params['N_HEADS'] = 2
    params['REBUILD_DATASET'] = True
    params['OPTIMIZED_SEARCH'] = True
    params['POS_UNK'] = False
    dataset = build_dataset(params)
    params['INPUT_VOCABULARY_SIZE'] = dataset.vocabulary_len[params['INPUTS_IDS_DATASET'][0]]
    params['OUTPUT_VOCABULARY_SIZE'] = dataset.vocabulary_len[params['OUTPUTS_IDS_DATASET'][0]]

    params['MODEL_NAME'] = \
        params['TASK_NAME'] + '_' + params['SRC_LAN'] + params['TRG_LAN'] + '_' + params['MODEL_TYPE'] + \
        '_model_size_' + str(params['MODEL_SIZE']) + \
        '_ff_size_' + str(params['FF_SIZE']) + \
        '_num_heads_' + str(params['N_HEADS']) + \
        '_encoder_blocks_' + str(params['N_LAYERS_ENCODER']) + \
        '_decoder_blocks_' + str(params['N_LAYERS_DECODER']) + \
        '_deepout_' + '_'.join([layer[0] for layer in params['DEEP_OUTPUT_LAYERS']]) + \
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
