# -*- coding: utf-8 -*-
# !/usr/bin/env python

from __future__ import print_function

try:
    import itertools.imap as map
except ImportError:
    pass
import argparse
import ast
import logging
import time
import sys
import os
import copy
import BaseHTTPServer
import urllib
from collections import OrderedDict
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../')
from keras_wrapper.model_ensemble import InteractiveBeamSearchSampler
from keras_wrapper.cnn_model import loadModel, updateModel
from keras_wrapper.dataset import loadDataset
from keras_wrapper.extra.isles_utils import *
from keras_wrapper.extra.read_write import pkl2dict, list2file
from keras_wrapper.online_trainer import OnlineTrainer
from keras_wrapper.utils import decode_predictions_beam_search, flatten_list_of_lists
from nmt_keras.model_zoo import TranslationModel
# from online_models import build_online_models
from utils.utils import update_parameters
from config_online import load_parameters as load_parameters_online
from config import load_parameters
logger = logging.getLogger(__name__)


class NMTHandler(BaseHTTPServer.BaseHTTPRequestHandler):
    def do_GET(self):
        do_GET_start_time = time.time()
        args = self.path.split('?')[1]
        args = args.split('&')
        source_sentence = None
        validated_prefix = None
        learn = False
        beam_size = 6
        length_norm = 0.
        coverage_norm = 0.
        alpha_norm = 1.
        args_processing_start_time = time.time()
        print (args)
        for aa in args:
            cc = aa.split('=')
            if cc[0] == 'source':
                source_sentence = urllib.unquote_plus(cc[1])

            if cc[0] == 'prefix':
                validated_prefix = cc[1]
                validated_prefix = urllib.unquote_plus(validated_prefix)

            if cc[0] == 'learn':
                learn = cc[1]
                learn = urllib.unquote_plus(learn)
                learn = eval(learn)

            if cc[0] == 'beam_size':
                beam_size = cc[1]
                beam_size = urllib.unquote_plus(beam_size)
                beam_size = int(beam_size)
                self.server.sampler.params_prediction['beam_size'] = beam_size

            if cc[0] == 'length_norm':
                length_norm = cc[1]
                length_norm = urllib.unquote_plus(length_norm)
                length_norm = float(length_norm)
                self.server.sampler.params_prediction['length_norm_factor'] = length_norm

            if cc[0] == 'coverage_norm':
                coverage_norm = cc[1]
                coverage_norm = urllib.unquote_plus(coverage_norm)
                coverage_norm = float(coverage_norm)
                self.server.sampler.params_prediction['coverage_norm_factor'] = coverage_norm

            if cc[0] == 'alpha_norm':
                alpha_norm = cc[1]
                alpha_norm = urllib.unquote_plus(alpha_norm)
                alpha_norm = float(alpha_norm)
                self.server.sampler.params_prediction['alpha_factor'] = alpha_norm

        if source_sentence is None:
            self.send_response(400)  # 400: ('Bad Request', 'Bad request syntax or unsupported method')
            return
        source_sentence = urllib.unquote_plus(source_sentence)
        args_processing_end_time = time.time()
        logger.log(2, 'args_processing time: %.6f' % (args_processing_end_time - args_processing_start_time))

        generate_sample_start_time = time.time()
        if learn and validated_prefix is not None and source_sentence is not None:
            self.server.sampler.learn_from_sample(source_sentence, validated_prefix)
            self.send_response(200)  # 200: ('OK', 'Request fulfilled, document follows')
        else:
            hypothesis = self.server.sampler.generate_sample(source_sentence, validated_prefix=validated_prefix)
            response = hypothesis + u'\n'
            generate_sample_end_time = time.time()
            logger.log(2, 'args_processing time: %.6f' % (generate_sample_end_time - generate_sample_start_time))
            send_response_start_time = time.time()
            self.send_response(200)  # 200: ('OK', 'Request fulfilled, document follows')
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(response.encode('utf-8'))
            send_response_end_time = time.time()
            logger.log(2, 'send_response time: %.6f' % (send_response_end_time - send_response_start_time))
            do_GET_end_time = time.time()
            logger.log(2, 'do_GET time: %.6f' % (do_GET_end_time - do_GET_start_time))


def parse_args():
    parser = argparse.ArgumentParser("Interactive neural machine translation server.")
    parser.add_argument("-ds", "--dataset", required=True, help="Dataset instance")
    parser.add_argument("-v", "--verbose", required=False, default=0, type=int, help="Verbosity level")
    parser.add_argument("-c", "--config", required=False, help="Config pkl for loading the model configuration. "
                                                               "If not specified, hyperparameters "
                                                               "are read from config.py")
    parser.add_argument("-m", "--models", nargs="+", required=True, help="Path to the models")
    parser.add_argument("-ch", "--changes", nargs="*", help="Changes to the config. Following the syntax Key=Value",
                        default="")
    parser.add_argument("-o", "--online",
                        action='store_true', default=False, required=False,
                        help="Online training mode after postedition. ")
    parser.add_argument("-a", "--address", help="Server address", type=str, default='')
    parser.add_argument("-p", "--port", help="Port to use", type=int, default=6542)
    parser.add_argument("-l", "--logging-level", help="Logging level: \t 0: Only info messages."
                                                      "\t 1: Debug messages."
                                                      "\t 2: Time monitoring messages.", type=int, default=0)
    parser.add_argument("-eos", "--eos-symbol", help="End-of-sentence symbol", type=str, default='/')

    return parser.parse_args()


class NMTSampler:
    def __init__(self, models, dataset, params, params_prediction, params_training, model_tokenize_f, model_detokenize_f, general_tokenize_f,
                 general_detokenize_f, mapping=None, word2index_x=None, word2index_y=None, index2word_y=None,
                 excluded_words=None, unk_id=1, eos_symbol='/', online=False, verbose=0):
        self.models = models
        self.dataset = dataset
        self.params = params
        self.params_prediction = params_prediction
        self.params_training = params_training
        self.model_tokenize_f = model_tokenize_f
        self.model_detokenize_f = model_detokenize_f
        self.general_tokenize_f = general_tokenize_f
        self.general_detokenize_f = general_detokenize_f
        self.mapping = mapping
        self.excluded_words = excluded_words
        self.verbose = verbose
        self.eos_symbol = eos_symbol
        self.word2index_x = word2index_x if word2index_x is not None else \
            dataset.vocabulary[params_prediction['INPUTS_IDS_DATASET'][0]]['words2idx']
        self.index2word_y = index2word_y if index2word_y is not None else \
            dataset.vocabulary[params_prediction['OUTPUTS_IDS_DATASET'][0]]['idx2words']
        self.word2index_y = word2index_y if word2index_y is not None else \
            dataset.vocabulary[params_prediction['OUTPUTS_IDS_DATASET'][0]]['words2idx']
        self.unk_id = unk_id
        self.interactive_beam_searcher = InteractiveBeamSearchSampler(self.models,
                                                                      self.dataset,
                                                                      self.params_prediction,
                                                                      excluded_words=self.excluded_words,
                                                                      verbose=self.verbose)

        # Compile Theano sampling function by generating a fake sample # TODO: Find a better way of doing this
        logger.info('Compiling sampler...')
        self.generate_sample('i')
        logger.info('Done.')

        self.online = online
        if self.online:
            self.online_trainer = OnlineTrainer(self.models, self.dataset, None,  # Sampler
                                                None,  # Params prediction
                                                params_training,
                                                verbose=self.verbose)
            for i, nmt_model in enumerate(self.models):
                logger.info('Compiling model %d...' % i)
                nmt_model.model._make_train_function()
            logger.info('Done.')

        else:
            self.online_trainer = None

    def generate_sample(self, source_sentence, validated_prefix=None, max_N=5, isle_indices=None,
                        filtered_idx2word=None, unk_indices=None, unk_words=None):
        print ("In params prediction beam_size: ", self.params_prediction['beam_size'])
        logger.log(2, 'Beam size: %d' % (self.params_prediction['beam_size']))
        generate_sample_start_time = time.time()
        if unk_indices is None:
            unk_indices = []
        if unk_words is None:
            unk_words = []

        tokenization_start_time = time.time()
        tokenized_input = self.general_tokenize_f(source_sentence, escape=False)
        tokenized_input = self.model_tokenize_f(tokenized_input)
        tokenization_end_time = time.time()
        logger.log(2, 'tokenization time: %.6f' % (tokenization_end_time - tokenization_start_time))
        parse_input_start_time = time.time()
        # Go from text to indices
        src_seq = self.dataset.loadText([tokenized_input],
                                        vocabularies=self.dataset.vocabulary[self.params['INPUTS_IDS_DATASET'][0]],
                                        max_len=self.params['MAX_INPUT_TEXT_LEN'],
                                        offset=0,
                                        fill=self.dataset.fill_text[self.params['INPUTS_IDS_DATASET'][0]],
                                        pad_on_batch=self.dataset.pad_on_batch[self.params['INPUTS_IDS_DATASET'][0]],
                                        words_so_far=False,
                                        loading_X=True)[0][0]

        parse_input_end_time = time.time()
        logger.log(2, 'parse_input time: %.6f' % (parse_input_end_time - parse_input_start_time))

        fixed_words_user = OrderedDict()
        unk_words_dict = OrderedDict()
        # If the user provided some feedback...
        if validated_prefix is not None:
            next_correction = validated_prefix[-1]
            if next_correction == self.eos_symbol:
                return validated_prefix[:-1].decode('utf-8')

            # 2.2.4 Tokenize the prefix properly (possibly applying BPE)
            #  TODO: Here we are tokenizing the target language with the source language tokenizer
            prefix_tokenization_start_time = time.time()
            tokenized_validated_prefix = self.general_tokenize_f(validated_prefix, escape=False)
            tokenized_validated_prefix = self.model_tokenize_f(tokenized_validated_prefix)
            prefix_tokenization_end_time = time.time()
            logger.log(2, 'prefix_tokenization time: %.6f' % (prefix_tokenization_end_time - prefix_tokenization_start_time))

            # 2.2.5 Validate words
            word_validation_start_time = time.time()
            for pos, word in enumerate(tokenized_validated_prefix.split()):
                fixed_words_user[pos] = self.word2index_y.get(word, self.unk_id)
                if self.word2index_y.get(word) is None:
                    unk_words_dict[pos] = word
            word_validation_end_time = time.time()
            logger.log(2, 'word_validation time: %.6f' % (word_validation_end_time - word_validation_start_time))

            # 2.2.6 Constrain search for the last word
            constrain_search_start_time = time.time()
            last_user_word_pos = list(fixed_words_user.keys())[-1]
            if next_correction != u' ':
                last_user_word = tokenized_validated_prefix.split()[-1]
                filtered_idx2word = dict((self.word2index_y[candidate_word], candidate_word)
                                         for candidate_word in self.word2index_y if candidate_word[:len(last_user_word)] == last_user_word)

                # if candidate_word.decode('utf-8')[:len(last_user_word)] == last_user_word)
                if filtered_idx2word != dict():
                    del fixed_words_user[last_user_word_pos]
                    if last_user_word_pos in list(unk_words_dict.keys()):
                        del unk_words_dict[last_user_word_pos]
            else:
                filtered_idx2word = dict()
            constrain_search_end_time = time.time()
            logger.log(2, 'constrain_search_end_time time: %.6f' % (constrain_search_end_time - constrain_search_start_time))

        sample_beam_search_start_time = time.time()
        trans_indices, costs, alphas = \
            self.interactive_beam_searcher.sample_beam_search_interactive(src_seq,
                                                                          fixed_words=copy.copy(fixed_words_user),
                                                                          max_N=max_N,
                                                                          isles=isle_indices,
                                                                          valid_next_words=filtered_idx2word,
                                                                          idx2word=self.index2word_y)
        sample_beam_search_end_time = time.time()
        logger.log(2, 'sample_beam_search time: %.6f' % (sample_beam_search_end_time - sample_beam_search_start_time))

        # # Substitute possible unknown words in isles
        # unk_in_isles = []
        # for isle_idx, isle_sequence, isle_words in unks_in_isles:
        #     if unk_id in isle_sequence:
        #         unk_in_isles.append((subfinder(isle_sequence, list(trans_indices)), isle_words))

        if False and self.params_prediction['pos_unk']:
            alphas = [alphas]
            sources = [tokenized_input]
            heuristic = self.params_prediction['heuristic']
        else:
            alphas = None
            heuristic = None
            sources = None

        # 1.2 Decode hypothesis
        decoding_predictions_start_time = time.time()
        hypothesis = decode_predictions_beam_search([trans_indices],
                                                    self.index2word_y,
                                                    alphas=alphas,
                                                    x_text=sources,
                                                    heuristic=heuristic,
                                                    mapping=self.mapping,
                                                    pad_sequences=True,
                                                    verbose=0)[0]
        decoding_predictions_end_time = time.time()
        logger.log(2, 'decoding_predictions time: %.6f' % (decoding_predictions_end_time - decoding_predictions_start_time))

        # for (words_idx, starting_pos), words in unk_in_isles:
        #     for pos_unk_word, pos_hypothesis in enumerate(range(starting_pos, starting_pos + len(words_idx))):
        #         hypothesis[pos_hypothesis] = words[pos_unk_word]

        # UNK words management
        unk_management_start_time = time.time()
        unk_indices = list(unk_words_dict)
        unk_words = list(unk_words_dict.values())
        if len(unk_indices) > 0:  # If we added some UNK word
            hypothesis = hypothesis.split()
            if len(hypothesis) < len(unk_indices):  # The full hypothesis will be made up UNK words:
                for i, index in enumerate(range(0, len(hypothesis))):
                    hypothesis[index] = unk_words[unk_indices[i]]
                for ii in range(i + 1, len(unk_words)):
                    hypothesis.append(unk_words[ii])
            else:  # We put each unknown word in the corresponding gap
                for i, index in enumerate(unk_indices):
                    if index < len(hypothesis):
                        hypothesis[index] = unk_words[i]
                    else:
                        hypothesis.append(unk_words[i])
            hypothesis = u' '.join(hypothesis)
        unk_management_end_time = time.time()
        logger.log(2, 'unk_management time: %.6f' % (unk_management_end_time - unk_management_start_time))

        hypothesis_detokenization_start_time = time.time()
        hypothesis = self.model_detokenize_f(hypothesis)
        hypothesis = self.general_detokenize_f(hypothesis, unescape=False)
        hypothesis_detokenization_end_time = time.time()
        logger.log(2, 'hypothesis_detokenization time: %.6f' % (hypothesis_detokenization_end_time - hypothesis_detokenization_start_time))
        generate_sample_end_time = time.time()
        logger.log(2, 'generate_sample time: %.6f' % (generate_sample_end_time - generate_sample_start_time))
        return hypothesis

    def learn_from_sample(self, source_sentence, target_sentence):

        # Tokenize input
        tokenized_input = self.general_tokenize_f(source_sentence, escape=False)
        tokenized_input = self.model_tokenize_f(tokenized_input)
        src_seq = self.dataset.loadText([tokenized_input],
                                        vocabularies=self.dataset.vocabulary[self.params['INPUTS_IDS_DATASET'][0]],
                                        max_len=self.params['MAX_INPUT_TEXT_LEN'],
                                        offset=0,
                                        fill=self.dataset.fill_text[self.params['INPUTS_IDS_DATASET'][0]],
                                        pad_on_batch=self.dataset.pad_on_batch[self.params['INPUTS_IDS_DATASET'][0]],
                                        words_so_far=False,
                                        loading_X=True)[0][0]
        # Tokenize output
        tokenized_reference = self.general_tokenize_f(target_sentence, escape=False)
        tokenized_reference = self.model_tokenize_f(tokenized_reference)

        # Build inputs/outpus of the system
        state_below = self.dataset.loadText([tokenized_reference.encode('utf-8')],
                                            vocabularies=self.dataset.vocabulary[self.params['OUTPUTS_IDS_DATASET'][0]],
                                            max_len=self.params['MAX_OUTPUT_TEXT_LEN_TEST'],
                                            offset=1,
                                            fill=self.dataset.fill_text[self.params['INPUTS_IDS_DATASET'][-1]],
                                            pad_on_batch=self.dataset.pad_on_batch[self.params['INPUTS_IDS_DATASET'][-1]],
                                            words_so_far=False,
                                            loading_X=True)[0]

        # 4.1.3 Ground truth sample -> Interactively translated sentence
        # TODO: Load dense-text if necessary
        trg_seq = self.dataset.loadTextOneHot([tokenized_reference.encode('utf-8')],
                                              vocabularies=self.dataset.vocabulary[self.params['OUTPUTS_IDS_DATASET'][0]],
                                              vocabulary_len=self.dataset.vocabulary_len[self.params['OUTPUTS_IDS_DATASET'][0]],
                                              max_len=self.params['MAX_OUTPUT_TEXT_LEN_TEST'],
                                              offset=0,
                                              fill=self.dataset.fill_text[self.params['OUTPUTS_IDS_DATASET'][0]],
                                              pad_on_batch=self.dataset.pad_on_batch[self.params['OUTPUTS_IDS_DATASET'][0]],
                                              words_so_far=False,
                                              sample_weights=self.params['SAMPLE_WEIGHTS'],
                                              loading_X=False)
        # 4.2 Train online!
        if self.online_trainer is not None:
            self.online_trainer.train_online([np.asarray([src_seq]), state_below], trg_seq, trg_words=[target_sentence])
        else:
            logging.warning('Online learning is disabled.')


def main():
    args = parse_args()
    server_address = (args.address, args.port)
    httpd = BaseHTTPServer.HTTPServer(server_address, NMTHandler)
    logger.setLevel(args.logging_level)
    parameters = load_parameters()
    if args.config is not None:
        logging.info("Loading parameters from %s" % str(args.config))
        parameters = update_parameters(parameters, pkl2dict(args.config))

    if args.online:
        online_parameters = load_parameters_online()
        parameters = update_parameters(parameters, online_parameters)

    try:
        for arg in args.changes:
            try:
                k, v = arg.split('=')
            except ValueError:
                print('Overwritten arguments must have the form key=Value. \n Currently are: %s' % str(args.changes))
                exit(1)
            try:
                parameters[k] = ast.literal_eval(v)
            except ValueError:
                parameters[k] = v
    except ValueError:
        print('Error processing arguments: (', k, ",", v, ")")
        exit(2)
    dataset = loadDataset(args.dataset)

    # For converting predictions into sentences
    # Dataset backwards compatibility
    bpe_separator = dataset.BPE_separator if hasattr(dataset,
                                                     "BPE_separator") and dataset.BPE_separator is not None else '@@'
    # Build BPE tokenizer if necessary
    if 'bpe' in parameters['TOKENIZATION_METHOD'].lower():
        logger.info('Building BPE')
        if not dataset.BPE_built:
            dataset.build_bpe(parameters.get('BPE_CODES_PATH', parameters['DATA_ROOT_PATH'] + '/training_codes.joint'),
                              bpe_separator)
    # Build tokenization function
    tokenize_f = eval('dataset.' + parameters.get('TOKENIZATION_METHOD', 'tokenize_bpe'))
    detokenize_function = eval('dataset.' + parameters.get('DETOKENIZATION_METHOD', 'detokenize_bpe'))
    dataset.build_moses_tokenizer(language=parameters['SRC_LAN'])
    dataset.build_moses_detokenizer(language=parameters['TRG_LAN'])
    tokenize_general = dataset.tokenize_moses
    detokenize_general = dataset.detokenize_moses

    params_prediction = dict()
    params_prediction['max_batch_size'] = parameters.get('BATCH_SIZE', 20)
    params_prediction['n_parallel_loaders'] = parameters.get('PARALLEL_LOADERS', 1)
    params_prediction['beam_size'] = parameters.get('BEAM_SIZE', 6)
    params_prediction['maxlen'] = parameters.get('MAX_OUTPUT_TEXT_LEN_TEST', 100)
    params_prediction['optimized_search'] = parameters['OPTIMIZED_SEARCH']
    params_prediction['model_inputs'] = parameters['INPUTS_IDS_MODEL']
    params_prediction['model_outputs'] = parameters['OUTPUTS_IDS_MODEL']
    params_prediction['dataset_inputs'] = parameters['INPUTS_IDS_DATASET']
    params_prediction['dataset_outputs'] = parameters['OUTPUTS_IDS_DATASET']
    params_prediction['search_pruning'] = parameters.get('SEARCH_PRUNING', False)
    params_prediction['normalize_probs'] = True
    params_prediction['alpha_factor'] = parameters.get('ALPHA_FACTOR', 1.0)
    params_prediction['coverage_penalty'] = True
    params_prediction['length_penalty'] = True
    params_prediction['length_norm_factor'] = parameters.get('LENGTH_NORM_FACTOR', 0.0)
    params_prediction['coverage_norm_factor'] = parameters.get('COVERAGE_NORM_FACTOR', 0.0)
    params_prediction['pos_unk'] = parameters.get('POS_UNK', False)
    params_prediction['heuristic'] = parameters.get('HEURISTIC', 0)

    params_prediction['state_below_maxlen'] = -1 if parameters.get('PAD_ON_BATCH', True) \
        else parameters.get('MAX_OUTPUT_TEXT_LEN', 50)
    params_prediction['output_max_length_depending_on_x'] = parameters.get('MAXLEN_GIVEN_X', True)
    params_prediction['output_max_length_depending_on_x_factor'] = parameters.get('MAXLEN_GIVEN_X_FACTOR', 3)
    params_prediction['output_min_length_depending_on_x'] = parameters.get('MINLEN_GIVEN_X', True)
    params_prediction['output_min_length_depending_on_x_factor'] = parameters.get('MINLEN_GIVEN_X_FACTOR', 2)
    params_prediction['attend_on_output'] = parameters.get('ATTEND_ON_OUTPUT', 'transformer' in parameters['MODEL_TYPE'].lower())

    # Manage pos_unk strategies
    if parameters['POS_UNK']:
        mapping = None if dataset.mapping == dict() else dataset.mapping
    else:
        mapping = None
    parameters_training = dict()
    if args.online:
        logging.info('Loading models from %s' % str(args.models))
        parameters_training = {  # Traning parameters
            'n_epochs': parameters['MAX_EPOCH'],
            'shuffle': False,
            'loss': parameters.get('LOSS', 'categorical_crossentropy'),
            'batch_size': parameters.get('BATCH_SIZE', 1),
            'homogeneous_batches': False,
            'optimizer': parameters.get('OPTIMIZER', 'SGD'),
            'lr': parameters.get('LR', 0.1),
            'lr_decay': parameters.get('LR_DECAY', None),
            'lr_gamma': parameters.get('LR_GAMMA', 1.),
            'epochs_for_save': -1,
            'verbose': args.verbose,
            'eval_on_sets': parameters['EVAL_ON_SETS_KERAS'],
            'n_parallel_loaders': parameters['PARALLEL_LOADERS'],
            'extra_callbacks': [],  # callbacks,
            'reload_epoch': parameters['RELOAD'],
            'epoch_offset': parameters['RELOAD'],
            'data_augmentation': parameters['DATA_AUGMENTATION'],
            'patience': parameters.get('PATIENCE', 0),
            'metric_check': parameters.get('STOP_METRIC', None),
            'eval_on_epochs': parameters.get('EVAL_EACH_EPOCHS', True),
            'each_n_epochs': parameters.get('EVAL_EACH', 1),
            'start_eval_on_epoch': parameters.get('START_EVAL_ON_EPOCH', 0),
            'additional_training_settings': {'k': parameters.get('K', 1),
                                             'tau': parameters.get('TAU', 1),
                                             'lambda': parameters.get('LAMBDA', 0.5),
                                             'c': parameters.get('C', 0.5),
                                             'd': parameters.get('D', 0.5)
                                             }
        }
        model_instances = [TranslationModel(parameters,
                                            model_type=parameters['MODEL_TYPE'],
                                            verbose=parameters['VERBOSE'],
                                            model_name=parameters['MODEL_NAME'] + '_' + str(i),
                                            vocabularies=dataset.vocabulary,
                                            store_path=parameters['STORE_PATH'],
                                            set_optimizer=False)
                           for i in range(len(args.models))]
        models = [updateModel(model, path, -1, full_path=True) for (model, path) in zip(model_instances, args.models)]

        # Set additional inputs to models if using a custom loss function
        # parameters['USE_CUSTOM_LOSS'] = True if 'PAS' in parameters['OPTIMIZER'] else False
        # if parameters.get('N_BEST_OPTIMIZER', False):
        #     logging.info('Using N-best optimizer')
        # models = build_online_models(models, parameters)
    else:
        models = [loadModel(m, -1, full_path=True) for m in args.models]

    for nmt_model in models:
        nmt_model.setParams(parameters)
        nmt_model.setOptimizer()

    parameters['INPUT_VOCABULARY_SIZE'] = dataset.vocabulary_len[parameters['INPUTS_IDS_DATASET'][0]]
    parameters['OUTPUT_VOCABULARY_SIZE'] = dataset.vocabulary_len[parameters['OUTPUTS_IDS_DATASET'][0]]

    # Get word2index and index2word dictionaries
    index2word_y = dataset.vocabulary[parameters['OUTPUTS_IDS_DATASET'][0]]['idx2words']
    word2index_y = dataset.vocabulary[parameters['OUTPUTS_IDS_DATASET'][0]]['words2idx']
    index2word_x = dataset.vocabulary[parameters['INPUTS_IDS_DATASET'][0]]['idx2words']
    word2index_x = dataset.vocabulary[parameters['INPUTS_IDS_DATASET'][0]]['words2idx']

    excluded_words = None
    interactive_beam_searcher = NMTSampler(models, dataset, parameters, params_prediction, parameters_training,
                                           tokenize_f, detokenize_function,
                                           tokenize_general, detokenize_general,
                                           mapping=mapping, word2index_x=word2index_x, word2index_y=word2index_y,
                                           index2word_y=index2word_y, eos_symbol=args.eos_symbol,
                                           excluded_words=excluded_words, online=args.online, verbose=args.verbose)

    httpd.sampler = interactive_beam_searcher

    logger.info('Server starting at %s' % str(server_address))
    httpd.serve_forever()


if __name__ == "__main__":
    main()
