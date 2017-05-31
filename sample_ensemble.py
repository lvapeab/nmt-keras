import argparse
import logging
import ast
from data_engine.prepare_data import update_dataset_from_file
from keras_wrapper.model_ensemble import BeamSearchEnsemble
from keras_wrapper.cnn_model import loadModel
from keras_wrapper.dataset import loadDataset
from keras_wrapper.extra.read_write import pkl2dict, list2file, nbest2file, list2stdout
from keras_wrapper.utils import decode_predictions_beam_search

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
    parser.add_argument("-m", "--models", nargs="+", required=True, help="Path to the models")
    parser.add_argument("-ch", "--changes", nargs="*", help="Changes to the config. Following the syntax Key=Value",
                        default="")
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()
    models = args.models
    logging.info("Using an ensemble of %d models" % len(args.models))
    models = [loadModel(m, -1, full_path=True) for m in args.models]
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
    heuristic = params.get('HEURISTIC', 0)
    mapping = None if dataset.mapping == dict() else dataset.mapping

    for s in args.splits:
        # Apply model predictions
        params_prediction['predict_on_sets'] = [s]
        beam_searcher = BeamSearchEnsemble(models, dataset, params_prediction,
                                           n_best=args.n_best, verbose=args.verbose)
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
            i = 0
            for i, (n_best_preds, n_best_scores, n_best_alphas) in enumerate(n_best):
                n_best_sample_score = []
                for n_best_pred, n_best_score, n_best_alpha in zip(n_best_preds, n_best_scores, n_best_alphas):
                    pred = decode_predictions_beam_search([n_best_pred],
                                                          index2word_y,
                                                          alphas=n_best_alpha,
                                                          x_text=sources,
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
