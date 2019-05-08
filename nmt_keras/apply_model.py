# -*- coding: utf-8 -*-
from __future__ import print_function
try:
    import itertools.imap as map
except ImportError:
    pass
import logging
from keras_wrapper.extra.read_write import list2file, nbest2file, list2stdout, numpy2file, pkl2dict

logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(__name__)


def sample_ensemble(args, params):
    """
    Use several translation models for obtaining predictions from a source text file.

    :param argparse.Namespace args: Arguments given to the method:

                      * dataset: Dataset instance with data.
                      * text: Text file with source sentences.
                      * splits: Splits to sample. Should be already included in the dataset object.
                      * dest: Output file to save scores.
                      * weights: Weight given to each model in the ensemble. You should provide the same number of weights than models. By default, it applies the same weight to each model (1/N).
                      * n_best: Write n-best list (n = beam size).
                      * config: Config .pkl for loading the model configuration. If not specified, hyperparameters are read from config.py.
                      * models: Path to the models.
                      * verbose: Be verbose or not.

    :param params: parameters of the translation model.
    """
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
    params_prediction['state_below_maxlen'] = -1 if params.get('PAD_ON_BATCH', True) \
        else params.get('MAX_OUTPUT_TEXT_LEN', 50)
    params_prediction['output_max_length_depending_on_x'] = params.get('MAXLEN_GIVEN_X', True)
    params_prediction['output_max_length_depending_on_x_factor'] = params.get('MAXLEN_GIVEN_X_FACTOR', 3)
    params_prediction['output_min_length_depending_on_x'] = params.get('MINLEN_GIVEN_X', True)
    params_prediction['output_min_length_depending_on_x_factor'] = params.get('MINLEN_GIVEN_X_FACTOR', 2)
    params_prediction['attend_on_output'] = params.get('ATTEND_ON_OUTPUT', 'transformer' in params['MODEL_TYPE'].lower())
    params_prediction['glossary'] = params.get('GLOSSARY', None)

    heuristic = params.get('HEURISTIC', 0)
    mapping = None if dataset.mapping == dict() else dataset.mapping
    model_weights = args.weights

    if args.glossary is not None:
        glossary = pkl2dict(args.glossary)
    elif params_prediction['glossary'] is not None:
        glossary = pkl2dict(params_prediction['glossary'])
    else:
        glossary = None

    if model_weights is not None and model_weights != []:
        assert len(model_weights) == len(models), 'You should give a weight to each model. You gave %d models and %d weights.' % (len(models), len(model_weights))
        model_weights = map(float, model_weights)
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
                                                     glossary=glossary,
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
                                                          glossary=glossary,
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


def score_corpus(args, params):
    """
    Use one or several translation models for scoring source--target pairs-

    :param argparse.Namespace args: Arguments given to the method:

                                * dataset: Dataset instance with data.
                                * source: Text file with source sentences.
                                * target: Text file with target sentences.
                                * splits: Splits to sample. Should be already included in the dataset object.
                                * dest: Output file to save scores.
                                * weights: Weight given to each model in the ensemble. You should provide the same number of weights than models. By default, it applies the same weight to each model (1/N).
                                * verbose: Be verbose or not.
                                * config: Config .pkl for loading the model configuration. If not specified, hyperparameters are read from config.py.
                                * models: Path to the models.
    :param dict params: parameters of the translation model.
    """

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
        model_weights = map(float, model_weights)
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
