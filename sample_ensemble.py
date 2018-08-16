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
    parser = argparse.ArgumentParser("Apply several translation models for making predictions")
    parser.add_argument("-ds", "--dataset", required=True, help="Dataset instance with data")
    parser.add_argument("-t", "--text", required=True, help="Text file with source sentences")
    parser.add_argument("-s", "--splits", nargs='+', required=False, default=['val'], help="Splits to sample. "
                                                                                           "Should be already included"
                                                                                           "into the dataset object.")
    parser.add_argument("-d", "--dest", required=False, help="File to save translations in. If not specified, "
                                                             "translations are outputted in STDOUT.")
    parser.add_argument("-v", "--verbose", required=False, default=0, type=int, help="Verbosity level")
    parser.add_argument("-c", "--config", required=False, help="Config pkl for loading the model configuration. "
                                                               "If not specified, hyperparameters "
                                                               "are read from config.py")
    parser.add_argument("-n", "--n-best", action="store_true", default=False, help="Write n-best list (n = beam size)")
    parser.add_argument("-w", "--weights", nargs="*", help="Weight given to each model in the ensemble. You should provide the same number of weights than models."
                                                           "By default, it applies the same weight to each model (1/N).", default=[])
    parser.add_argument("-m", "--models", nargs="+", required=True, help="Path to the models")
    parser.add_argument("-ch", "--changes", nargs="*", help="Changes to the config. Following the syntax Key=Value",
                        default="")
    return parser.parse_args()


def sample_ensemble(args, params):

    from data_engine.prepare_data import update_dataset_from_file
    from keras_wrapper.model_ensemble import BeamSearchEnsemble
    from keras_wrapper.cnn_model import loadModel
    from keras_wrapper.dataset import loadDataset
    from keras_wrapper.utils import decode_predictions_beam_search

    logging.info("Using an ensemble of %d models" % len(args.models))
    models = [loadModel(m, -1, full_path=True) for m in args.models]
    dataset = loadDataset(args.dataset)
    dataset = update_dataset_from_file(dataset, args.text, params, splits=args.splits, remove_outputs=True)

    params['INPUT_VOCABULARY_SIZE'] = dataset.vocabulary_len[params['INPUTS_IDS_DATASET'][0]]
    params['OUTPUT_VOCABULARY_SIZE'] = dataset.vocabulary_len[params['OUTPUTS_IDS_DATASET'][0]]
    # For converting predictions into sentences
    index2word_y = dataset.vocabulary[params['OUTPUTS_IDS_DATASET'][0]]['idx2words']

    if params.get('APPLY_DETOKENIZATION', False):
        detokenize_function = eval('dataset.' + params['DETOKENIZATION_METHOD'])

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
    params_prediction['state_below_maxlen'] = -1 if params.get('PAD_ON_BATCH', True) else params.get('MAX_OUTPUT_TEXT_LEN', 50)
    params_prediction['output_max_length_depending_on_x'] = params.get('MAXLEN_GIVEN_X', True)
    params_prediction['output_max_length_depending_on_x_factor'] = params.get('MAXLEN_GIVEN_X_FACTOR', 3)
    params_prediction['output_min_length_depending_on_x'] = params.get('MINLEN_GIVEN_X', True)
    params_prediction['output_min_length_depending_on_x_factor'] = params.get('MINLEN_GIVEN_X_FACTOR', 2)
    params_prediction['attend_on_output'] = params.get('ATTEND_ON_OUTPUT', 'transformer' in params['MODEL_TYPE'].lower())

    heuristic = params.get('HEURISTIC', 0)
    mapping = None if dataset.mapping == dict() else dataset.mapping
    model_weights = args.weights

    if model_weights is not None and model_weights != []:
        assert len(model_weights) == len(models), 'You should give a weight to each model. You gave %d models and %d weights.' % (len(models), len(model_weights))
        model_weights = map(lambda x: float(x), model_weights)
        if len(model_weights) > 1:
            logger.info('Giving the following weights to each model: %s' % str(model_weights))
    for s in args.splits:
        # Apply model predictions
        params_prediction['predict_on_sets'] = [s]
        beam_searcher = BeamSearchEnsemble(models, dataset, params_prediction,
                                           model_weights=model_weights, n_best=args.n_best, verbose=args.verbose)
        if args.n_best:
            predictions, n_best = beam_searcher.predictBeamSearchNet()[s]
        else:
            predictions = beam_searcher.predictBeamSearchNet()[s]
            n_best = None
        if params_prediction['pos_unk']:
            samples = predictions[0]
            alphas = predictions[1]
            sources = [x.strip() for x in open(args.text, 'r').read().split('\n')]
            sources = sources[:-1] if len(sources[-1]) == 0 else sources
        else:
            samples = predictions
            alphas = None
            heuristic = None
            sources = None

        predictions = decode_predictions_beam_search(samples,
                                                     index2word_y,
                                                     alphas=alphas,
                                                     x_text=sources,
                                                     heuristic=heuristic,
                                                     mapping=mapping,
                                                     verbose=args.verbose)
        # Apply detokenization function if needed
        if params.get('APPLY_DETOKENIZATION', False):
            predictions = map(detokenize_function, predictions)

        if args.n_best:
            n_best_predictions = []
            for i, (n_best_preds, n_best_scores, n_best_alphas) in enumerate(n_best):
                n_best_sample_score = []
                for n_best_pred, n_best_score, n_best_alpha in zip(n_best_preds, n_best_scores, n_best_alphas):
                    pred = decode_predictions_beam_search([n_best_pred],
                                                          index2word_y,
                                                          alphas=[n_best_alpha] if params_prediction['pos_unk'] else None,
                                                          x_text=[sources[i]] if params_prediction['pos_unk'] else None,
                                                          heuristic=heuristic,
                                                          mapping=mapping,
                                                          verbose=args.verbose)
                    # Apply detokenization function if needed
                    if params.get('APPLY_DETOKENIZATION', False):
                        pred = map(detokenize_function, pred)

                    n_best_sample_score.append([i, pred, n_best_score])
                n_best_predictions.append(n_best_sample_score)
        # Store result
        if args.dest is not None:
            filepath = args.dest  # results file
            if params.get('SAMPLING_SAVE_MODE', 'list'):
                list2file(filepath, predictions)
                if args.n_best:
                    nbest2file(filepath + '.nbest', n_best_predictions)
            else:
                raise Exception('Only "list" is allowed in "SAMPLING_SAVE_MODE"')
        else:
            list2stdout(predictions)
            if args.n_best:
                logging.info('Storing n-best sentences in ./' + s + '.nbest')
                nbest2file('./' + s + '.nbest', n_best_predictions)
        logging.info('Sampling finished')


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
    sample_ensemble(args, params)
