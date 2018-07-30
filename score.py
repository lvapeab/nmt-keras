# -*- coding: utf-8 -*-
from __future__ import print_function
try:
    import itertools.imap as map
except ImportError:
    pass
import argparse
import logging
import ast
from keras_wrapper.extra.read_write import pkl2dict, list2file, nbest2file, list2stdout

logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser("Use several translation models for scoring source--target pairs")
    parser.add_argument("-ds", "--dataset", required=True, help="Dataset instance with data")
    parser.add_argument("-src", "--source", required=True, help="Text file with source sentences")
    parser.add_argument("-trg", "--target", required=True, help="Text file with target sentences")
    parser.add_argument("-s", "--splits", nargs='+', required=False, default=['val'], help="Splits to sample. "
                                                                                           "Should be already included"
                                                                                           "into the dataset object.")
    parser.add_argument("-d", "--dest", required=False, help="File to save scores in")
    parser.add_argument("-v", "--verbose", required=False, action='store_true', default=False, help="Be verbose")
    parser.add_argument("-w", "--weights", nargs="*", help="Weight given to each model in the ensemble. "
                                                           "You should provide the same number of weights than models."
                                                           "By default, it applies the same weight to each model (1/N).", default=[])
    parser.add_argument("-c", "--config", required=False, help="Config pkl for loading the model configuration. "
                                                               "If not specified, hyperparameters "
                                                               "are read from config.py")
    parser.add_argument("--models", nargs='+', required=True, help="path to the models")
    parser.add_argument("-ch", "--changes", nargs="*", help="Changes to the config. Following the syntax Key=Value",
                        default="")
    return parser.parse_args()


def score_corpus(args, params):

    from data_engine.prepare_data import update_dataset_from_file
    from keras_wrapper.dataset import loadDataset
    from keras_wrapper.cnn_model import loadModel
    from keras_wrapper.model_ensemble import BeamSearchEnsemble

    logging.info("Using an ensemble of %d models" % len(args.models))
    models = [loadModel(m, -1, full_path=True) for m in args.models]
    dataset = loadDataset(args.dataset)
    dataset = update_dataset_from_file(dataset, args.source, params, splits=args.splits,
                                       output_text_filename=args.target, compute_state_below=True)

    params['INPUT_VOCABULARY_SIZE'] = dataset.vocabulary_len[params['INPUTS_IDS_DATASET'][0]]
    params['OUTPUT_VOCABULARY_SIZE'] = dataset.vocabulary_len[params['OUTPUTS_IDS_DATASET'][0]]
    # Apply scoring
    extra_vars = dict()
    extra_vars['tokenize_f'] = eval('dataset.' + params['TOKENIZATION_METHOD'])

    model_weights = args.weights
    if model_weights is not None and model_weights != []:
        assert len(model_weights) == len(models), 'You should give a weight to each model. You gave %d models and %d weights.' % (len(models), len(model_weights))
        model_weights = map(lambda x: float(x), model_weights)
        if len(model_weights) > 1:
            logger.info('Giving the following weights to each model: %s' % str(model_weights))

    for s in args.splits:
        # Apply model predictions
        params_prediction = {'max_batch_size': params['BATCH_SIZE'],
                             'n_parallel_loaders': params['PARALLEL_LOADERS'],
                             'predict_on_sets': [s]}

        if params['BEAM_SEARCH']:
            params_prediction['beam_size'] = params['BEAM_SIZE']
            params_prediction['maxlen'] = params['MAX_OUTPUT_TEXT_LEN_TEST']
            params_prediction['optimized_search'] = params['OPTIMIZED_SEARCH']
            params_prediction['model_inputs'] = params['INPUTS_IDS_MODEL']
            params_prediction['model_outputs'] = params['OUTPUTS_IDS_MODEL']
            params_prediction['dataset_inputs'] = params['INPUTS_IDS_DATASET']
            params_prediction['dataset_outputs'] = params['OUTPUTS_IDS_DATASET']
            params_prediction['normalize_probs'] = params.get('NORMALIZE_SAMPLING', False)
            params_prediction['alpha_factor'] = params.get('ALPHA_FACTOR', 1.0)
            params_prediction['coverage_penalty'] = params.get('COVERAGE_PENALTY', False)
            params_prediction['length_penalty'] = params.get('LENGTH_PENALTY', False)
            params_prediction['length_norm_factor'] = params.get('LENGTH_NORM_FACTOR', 0.0)
            params_prediction['coverage_norm_factor'] = params.get('COVERAGE_NORM_FACTOR', 0.0)
            params_prediction['pos_unk'] = params.get('POS_UNK', False)
            params_prediction['state_below_maxlen'] = -1 if params.get('PAD_ON_BATCH', True) \
                else params.get('MAX_OUTPUT_TEXT_LEN', 50)
            params_prediction['output_max_length_depending_on_x'] = params.get('MAXLEN_GIVEN_X', True)
            params_prediction['output_max_length_depending_on_x_factor'] = params.get('MAXLEN_GIVEN_X_FACTOR', 3)
            params_prediction['output_min_length_depending_on_x'] = params.get('MINLEN_GIVEN_X', True)
            params_prediction['output_min_length_depending_on_x_factor'] = params.get('MINLEN_GIVEN_X_FACTOR', 2)
            params_prediction['attend_on_output'] = params.get('ATTEND_ON_OUTPUT', 'transformer' in params['MODEL_TYPE'].lower())
            beam_searcher = BeamSearchEnsemble(models, dataset, params_prediction, model_weights=model_weights, verbose=args.verbose)
            scores = beam_searcher.scoreNet()[s]

        # Store result
        if args.dest is not None:
            filepath = args.dest  # results file
            if params['SAMPLING_SAVE_MODE'] == 'list':
                list2file(filepath, scores)
            elif params['SAMPLING_SAVE_MODE'] == 'numpy':
                numpy2file(filepath, scores)
            else:
                raise Exception('The sampling mode ' + params['SAMPLING_SAVE_MODE'] + ' is not currently supported.')
        else:
            print (scores)


def check_params(params):
    """
    Checks some typical parameters and warns if something wrong was specified.
    :param params: Model instance on which to apply the callback.
    :return: None
    """

    if params['SRC_PRETRAINED_VECTORS'] and params['SRC_PRETRAINED_VECTORS'][:-1] != '.npy':
        logger.warn('It seems that the pretrained word vectors provided for the target text are not in npy format.'
                    'You should preprocess the word embeddings with the "utils/preprocess_*_word_vectors.py script.')

    if params['TRG_PRETRAINED_VECTORS'] and params['TRG_PRETRAINED_VECTORS'][:-1] != '.npy':
        logger.warn('It seems that the pretrained word vectors provided for the target text are not in npy format.'
                    'You should preprocess the word embeddings with the "utils/preprocess_*_word_vectors.py script.')
    if not params['PAD_ON_BATCH']:
        logger.warn('It is HIGHLY recommended to set the option "PAD_ON_BATCH = True."')

    if params['MODEL_TYPE'].lower() == 'transformer':

        assert params['MODEL_SIZE'] == params['TARGET_TEXT_EMBEDDING_SIZE'], 'When using the Transformer model, ' \
                                                                             'dimensions of "MODEL_SIZE" and "TARGET_TEXT_EMBEDDING_SIZE" must match. ' \
                                                                             'Currently, they are: %d and %d, respectively.' % (params['MODEL_SIZE'], params['TARGET_TEXT_EMBEDDING_SIZE'])
        assert params['MODEL_SIZE'] == params['SOURCE_TEXT_EMBEDDING_SIZE'], 'When using the Transformer model, ' \
                                                                             'dimensions of "MODEL_SIZE" and "SOURCE_TEXT_EMBEDDING_SIZE" must match. ' \
                                                                             'Currently, they are: %d and %d, respectively.' % (params['MODEL_SIZE'], params['SOURCE_TEXT_EMBEDDING_SIZE'])

        if params['POS_UNK']:
            logger.warn('The "POS_UNK" option is still unimplemented for the "Transformer" model. Setting it to False.')
            params['POS_UNK'] = False
        assert params['MODEL_SIZE'] % params['N_HEADS'] == 0, \
            '"MODEL_SIZE" should be a multiple of "N_HEADS". ' \
            'Currently: mod(%d, %d) == %d.' % (params['MODEL_SIZE'], params['N_HEADS'], params['MODEL_SIZE'] % params['N_HEADS'])

    if params['POS_UNK']:
        if not params['OPTIMIZED_SEARCH']:
            logger.warn('Unknown words replacement requires to use the optimized search ("OPTIMIZED_SEARCH" parameter). Setting "POS_UNK" to False.')
            params['POS_UNK'] = False

    if params['COVERAGE_PENALTY']:
        assert params['OPTIMIZED_SEARCH'], 'The application of "COVERAGE_PENALTY" requires ' \
                                           'to use the optimized search ("OPTIMIZED_SEARCH" parameter).'
    return params


if __name__ == "__main__":

    args = parse_args()
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
                print ('Overwritten arguments must have the form key=Value. \n Currently are: %s' % str(args.changes))
                exit(1)
            try:
                params[k] = ast.literal_eval(v)
            except ValueError:
                params[k] = v
    except ValueError:
        print ('Error processing arguments: (', k, ",", v, ")")
        exit(2)
    params = check_params(params)
    score_corpus(args, params)
