#!/usr/bin/env python
import argparse
import cPickle
import traceback
import logging
import time
import sys
import os
import copy
import numpy
import BaseHTTPServer
import urllib
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../')
from collections import OrderedDict
from config import load_parameters
from data_engine.prepare_data import update_dataset_from_file
from keras_wrapper.beam_search_interactive import InteractiveBeamSearchSampler
from keras_wrapper.cnn_model import loadModel, updateModel
from keras_wrapper.dataset import loadDataset
from keras_wrapper.extra.isles_utils import *
from keras_wrapper.extra.read_write import pkl2dict, list2file
from keras_wrapper.online_trainer import OnlineTrainer
from keras_wrapper.utils import decode_predictions_beam_search, flatten_list_of_lists
from subprocess import Popen, PIPE
from DocXMLRPCServer import DocXMLRPCServer

logger = logging.getLogger(__name__)


class NMTHandler(BaseHTTPServer.BaseHTTPRequestHandler):
    def do_GET(self):
        args = self.path.split('?')[1]
        args = args.split('&')
        source_sentence = None
        for aa in args:
            cc = aa.split('=')
            if cc[0] == 'source':
                source_sentence = cc[1]
            if cc[0] == 'prefix':
                validated_prefix = cc[1]
                validated_prefix = urllib.unquote_plus(validated_prefix)

            else:
                validated_prefix = None
        if source_sentence is None:
            self.send_response(400)
            return

        source_sentence = urllib.unquote_plus(source_sentence)

        hypothesis = self.server.sampler.generate_sample(source_sentence,
                                                         validated_prefix=validated_prefix)

        response = hypothesis + u'\n'
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(response.encode('utf-8'))


def parse_args():
    parser = argparse.ArgumentParser("Interactive neural machine translation server.")
    parser.add_argument("-ds", "--dataset", required=True, help="Dataset instance")
    parser.add_argument("-v", "--verbose", required=False, default=0, type=int, help="Verbosity level")
    parser.add_argument("-c", "--config", required=False, help="Config pkl for loading the model configuration. "
                                                               "If not specified, hyperparameters "
                                                               "are read from config.py")
    parser.add_argument("-n", "--n-best", action="store_true", default=False, help="Write n-best list (n = beam size)")
    parser.add_argument("-m", "--models", nargs="+", required=True, help="Path to the models")
    parser.add_argument("-ch", "--changes", nargs="*", help="Changes to the config. Following the syntax Key=Value",
                        default="")
    parser.add_argument("-o", "--online",
                        action='store_true', default=False, required=False,
                        help="Online training mode after postedition. ")
    parser.add_argument("-p", "--port", help="Port to use", type=int, default=8888)

    return parser.parse_args()


class NMTSampler:
    def __init__(self, models, dataset, params_prediction, model_tokenize_f, model_detokenize_f, general_tokenize_f,
                 general_detokenize_f, mapping=None, word2index_x=None, word2index_y=None, index2word_y=None,
                 excluded_words=None, unk_id=1,
                 verbose=0):
        self.models = models
        self.dataset = dataset
        self.params_prediction = params_prediction
        self.model_tokenize_f = model_tokenize_f
        self.model_detokenize_f = model_detokenize_f
        self.general_tokenize_f = general_tokenize_f
        self.general_detokenize_f = general_detokenize_f
        self.mapping = mapping
        self.excluded_words = excluded_words
        self.verbose = verbose
        self.word2index_x = word2index_x if word2index_x is not None else \
            dataset.vocabulary[params_prediction['INPUTS_IDS_DATASET'][0]]['words2idx']
        self.index2word_y = index2word_y if index2word_y is not None else \
            dataset.vocabulary[params['OUTPUTS_IDS_DATASET'][0]]['idx2words']
        self.word2index_y = word2index_y if word2index_y is not None else \
            dataset.vocabulary[params['OUTPUTS_IDS_DATASET'][0]]['words2idx']
        self.unk_id = unk_id
        self.interactive_beam_searcher = InteractiveBeamSearchSampler(self.models,
                                                                      self.dataset,
                                                                      self.params_prediction,
                                                                      excluded_words=self.excluded_words,
                                                                      verbose=self.verbose)

    def generate_sample(self, source_sentence, validated_prefix=None, max_N=5, isle_indices=None,
                        filtered_idx2word=None, unk_indices=None, unk_words=None):

        if unk_indices is None:
            unk_indices = []
        if unk_words is None:
            unk_words = []

        tokenized_input = self.general_tokenize_f(source_sentence)
        tokenized_input = self.model_tokenize_f(tokenized_input)
        src_seq, src_words = parse_input(tokenized_input, self.dataset, self.word2index_x)
        fixed_words_user = OrderedDict()
        unk_words_dict = OrderedDict()
        # If the user provided some feedback...
        if validated_prefix is not None:
            next_correction = validated_prefix[-1]

            # 2.2.4 Tokenize the prefix properly (possibly applying BPE)
            #  TODO: Here we are tokenizing the target language with the source language tokenizer
            tokenized_validated_prefix = self.general_tokenize_f(validated_prefix)
            tokenized_validated_prefix = self.model_tokenize_f(tokenized_validated_prefix)

            # 2.2.5 Validate words
            for pos, word in enumerate(tokenized_validated_prefix.split()):
                fixed_words_user[pos] = self.word2index_y.get(word, self.unk_id)
                if self.word2index_y.get(word) is None:
                    unk_words_dict[pos] = word
            # 2.2.6 Constrain search for the last word
            last_user_word_pos = fixed_words_user.keys()[-1]
            if next_correction != u' ':
                last_user_word = tokenized_validated_prefix.split()[-1]
                filtered_idx2word = dict((self.word2index_y[candidate_word], candidate_word)
                                         for candidate_word in self.word2index_y
                                         if candidate_word.decode('utf-8')[:len(last_user_word)]
                                         == last_user_word)
                if filtered_idx2word != dict():
                    del fixed_words_user[last_user_word_pos]
                    if last_user_word_pos in unk_words_dict.keys():
                        del unk_words_dict[last_user_word_pos]
            else:
                filtered_idx2word = dict()

        trans_indices, costs, alphas = \
            self.interactive_beam_searcher.sample_beam_search_interactive(src_seq,
                                                                          fixed_words=copy.copy(fixed_words_user),
                                                                          max_N=max_N,
                                                                          isles=isle_indices,
                                                                          valid_next_words=filtered_idx2word,
                                                                          idx2word=self.index2word_y)
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
        hypothesis = decode_predictions_beam_search([trans_indices],
                                                    self.index2word_y,
                                                    alphas=alphas,
                                                    x_text=sources,
                                                    heuristic=heuristic,
                                                    mapping=self.mapping,
                                                    pad_sequences=True,
                                                    verbose=0)[0]

        # for (words_idx, starting_pos), words in unk_in_isles:
        #     for pos_unk_word, pos_hypothesis in enumerate(range(starting_pos, starting_pos + len(words_idx))):
        #         hypothesis[pos_hypothesis] = words[pos_unk_word]

        # UNK words management
        unk_indices = unk_words_dict.keys()
        unk_words = unk_words_dict.values()
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

        hypothesis = self.model_detokenize_f(hypothesis)
        hypothesis = self.general_detokenize_f(hypothesis)
        return hypothesis


def main():
    args = parse_args()
    server_address = ('', args.port)
    httpd = BaseHTTPServer.HTTPServer(server_address, NMTHandler)

    if args.config is None:
        logging.info("Reading parameters from config.py")
        from config import load_parameters
        params = load_parameters()
    else:
        logging.info("Loading parameters from %s" % str(args.config))
        params = pkl2dict(args.config)
    try:
        for arg in args.changes:
            try:
                k, v = arg.split('=')
            except ValueError:
                print 'Overwritten arguments must have the form key=Value. \n Currently are: %s' % str(args.changes)
                exit(1)
            try:
                params[k] = ast.literal_eval(v)
            except ValueError:
                params[k] = v
    except ValueError:
        print 'Error processing arguments: (', k, ",", v, ")"
        exit(2)
    dataset = loadDataset(args.dataset)

    # For converting predictions into sentences
    # Dataset backwards compatibility
    bpe_separator = dataset.BPE_separator if hasattr(dataset,
                                                     "BPE_separator") and dataset.BPE_separator is not None else '@@'
    # Build BPE tokenizer if necessary
    if 'bpe' in params['TOKENIZATION_METHOD'].lower():
        logger.info('Building BPE')
        if not dataset.BPE_built:
            dataset.build_bpe(params.get('BPE_CODES_PATH', params['DATA_ROOT_PATH'] + '/training_codes.joint'),
                              bpe_separator)
    # Build tokenization function
    tokenize_f = eval('dataset.' + params.get('TOKENIZATION_METHOD', 'tokenize_none'))

    detokenize_function = eval('dataset.' + params.get('DETOKENIZATION_METHOD', 'detokenize_none'))
    dataset.build_moses_tokenizer(language=params['SRC_LAN'])
    dataset.build_moses_detokenizer(language=params['TRG_LAN'])
    tokenize_general = dataset.tokenize_moses
    detokenize_general = dataset.detokenize_moses

    params_prediction = dict()
    params_prediction['max_batch_size'] = params.get('BATCH_SIZE', 20)
    params_prediction['n_parallel_loaders'] = params.get('PARALLEL_LOADERS', 1)
    params_prediction['beam_size'] = params.get('BEAM_SIZE', 6)
    params_prediction['maxlen'] = params.get('MAX_OUTPUT_TEXT_LEN_TEST', 100)
    params_prediction['optimized_search'] = params['OPTIMIZED_SEARCH']
    params_prediction['model_inputs'] = params['INPUTS_IDS_MODEL']
    params_prediction['model_outputs'] = params['OUTPUTS_IDS_MODEL']
    params_prediction['dataset_inputs'] = params['INPUTS_IDS_DATASET']
    params_prediction['dataset_outputs'] = params['OUTPUTS_IDS_DATASET']
    params_prediction['search_pruning'] = params.get('SEARCH_PRUNING', False)
    params_prediction['normalize_probs'] = params.get('NORMALIZE_SAMPLING', False)
    params_prediction['alpha_factor'] = params.get('ALPHA_FACTOR', 1.0)
    params_prediction['coverage_penalty'] = params.get('COVERAGE_PENALTY', False)
    params_prediction['length_penalty'] = params.get('LENGTH_PENALTY', False)
    params_prediction['length_norm_factor'] = params.get('LENGTH_NORM_FACTOR', 0.0)
    params_prediction['coverage_norm_factor'] = params.get('COVERAGE_NORM_FACTOR', 0.0)
    params_prediction['pos_unk'] = params.get('POS_UNK', False)
    params_prediction['heuristic'] = params.get('HEURISTIC', 0)

    params_prediction['state_below_maxlen'] = -1 if params.get('PAD_ON_BATCH', True) \
        else params.get('MAX_OUTPUT_TEXT_LEN', 50)
    params_prediction['output_max_length_depending_on_x'] = params.get('MAXLEN_GIVEN_X', True)
    params_prediction['output_max_length_depending_on_x_factor'] = params.get('MAXLEN_GIVEN_X_FACTOR', 3)
    params_prediction['output_min_length_depending_on_x'] = params.get('MINLEN_GIVEN_X', True)
    params_prediction['output_min_length_depending_on_x_factor'] = params.get('MINLEN_GIVEN_X_FACTOR', 2)
    # Manage pos_unk strategies
    if params['POS_UNK']:
        mapping = None if dataset.mapping == dict() else dataset.mapping
    else:
        mapping = None

    if args.online:
        logging.info('Loading models from %s' % str(args.models))

        model_instances = [TranslationModel(params,
                                            model_type=params['MODEL_TYPE'],
                                            verbose=params['VERBOSE'],
                                            model_name=params['MODEL_NAME'] + '_' + str(i),
                                            vocabularies=dataset.vocabulary,
                                            store_path=params['STORE_PATH'],
                                            set_optimizer=False)
                           for i in range(len(args.models))]
        models = [updateModel(model, path, -1, full_path=True) for (model, path) in zip(model_instances, args.models)]

        # Set additional inputs to models if using a custom loss function
        params['USE_CUSTOM_LOSS'] = True if 'PAS' in params['OPTIMIZER'] else False
        if params['N_BEST_OPTIMIZER']:
            logging.info('Using N-best optimizer')

        models = build_online_models(models, params)
        online_trainer = OnlineTrainer(models,
                                       dataset,
                                       None,
                                       None,
                                       params_training,
                                       verbose=args.verbose)
    else:
        models = [loadModel(m, -1, full_path=True) for m in args.models]

    params['INPUT_VOCABULARY_SIZE'] = dataset.vocabulary_len[params['INPUTS_IDS_DATASET'][0]]
    params['OUTPUT_VOCABULARY_SIZE'] = dataset.vocabulary_len[params['OUTPUTS_IDS_DATASET'][0]]

    # Get word2index and index2word dictionaries
    index2word_y = dataset.vocabulary[params['OUTPUTS_IDS_DATASET'][0]]['idx2words']
    word2index_y = dataset.vocabulary[params['OUTPUTS_IDS_DATASET'][0]]['words2idx']
    index2word_x = dataset.vocabulary[params['INPUTS_IDS_DATASET'][0]]['idx2words']
    word2index_x = dataset.vocabulary[params['INPUTS_IDS_DATASET'][0]]['words2idx']

    excluded_words = None
    interactive_beam_searcher = NMTSampler(models, dataset, params_prediction,
                                           tokenize_f, detokenize_function,
                                           tokenize_general, detokenize_general,
                                           mapping=mapping, word2index_x=word2index_x, word2index_y=word2index_y,
                                           index2word_y=index2word_y,
                                           excluded_words=excluded_words, verbose=args.verbose)

    # Compile Theano sampling function by generating a fake sample # TODO: Find a better way of doing this
    print "Compiling sampler..."
    interactive_beam_searcher.generate_sample('i')

    httpd.sampler = interactive_beam_searcher

    print 'Server starting at localhost:' + str(args.port)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
