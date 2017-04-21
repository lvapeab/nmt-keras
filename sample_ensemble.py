import argparse
import logging

from config import load_parameters
from data_engine.prepare_data import update_dataset_from_file, keep_n_captions
from keras_wrapper.beam_search_ensemble import BeamSearchEnsemble
from keras_wrapper.cnn_model import loadModel
from keras_wrapper.dataset import loadDataset
from keras_wrapper.extra.evaluation import select as evaluation_select
from keras_wrapper.extra.read_write import pkl2dict, list2file, nbest2file
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
    parser.add_argument("-d", "--dest", required=False, help="File to save translations in")
    parser.add_argument("--not-eval", action='store_true', default=False, help="Do not compute metrics for the output")
    parser.add_argument("-e", "--eval-output", required=False, help="Write evaluation results to file")
    parser.add_argument("-v", "--verbose", required=False, default=0, type=int, help="Verbosity level")
    parser.add_argument("-c", "--config", required=False, help="Config pkl for loading the model configuration. "
                                                               "If not specified, hyperparameters "
                                                               "are read from config.py")
    parser.add_argument("--n-best", action="store_true", default=False, help="Write n-best list (n = beam size)")
    parser.add_argument("--models", nargs='+', required=True, help="path to the models")
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    models = args.models
    print "Using an ensemble of %d models" % len(args.models)
    models = [loadModel(m, -1, full_path=True) for m in args.models]
    if args.config is None:
        print "Reading parameters from config.py"
        params = load_parameters()
    else:
        print "Loading parameters from %s" % str(args.config)
        params = pkl2dict(args.config)

    dataset = loadDataset(args.dataset)
    dataset = update_dataset_from_file(dataset, args.text, params, splits=args.splits, remove_outputs=args.not_eval)

    if args.eval_output:
        keep_n_captions(dataset, repeat=1, n=1, set_names=args.splits)  # Include extra variables (references)

    params['INPUT_VOCABULARY_SIZE'] = dataset.vocabulary_len[params['INPUTS_IDS_DATASET'][0]]
    params['OUTPUT_VOCABULARY_SIZE'] = dataset.vocabulary_len[params['OUTPUTS_IDS_DATASET'][0]]
    if params.get('APPLY_DETOKENIZATION', False):
        detokenize_function = eval('dataset.' + params['DETOKENIZATION_METHOD'])

    # Apply sampling
    extra_vars = dict()
    extra_vars['tokenize_f'] = eval('dataset.' + params['TOKENIZATION_METHOD'])
    for s in args.splits:
        # Apply model predictions
        params_prediction = {'batch_size': params['BATCH_SIZE'],
                             'n_parallel_loaders': params['PARALLEL_LOADERS'],
                             'predict_on_sets': [s]}

        # Convert predictions into sentences
        index2word_y = dataset.vocabulary[params['OUTPUTS_IDS_DATASET'][0]]['idx2words']

        if params['BEAM_SEARCH']:
            params_prediction['beam_size'] = params['BEAM_SIZE']
            params_prediction['maxlen'] = params['MAX_OUTPUT_TEXT_LEN_TEST']
            params_prediction['optimized_search'] = params['OPTIMIZED_SEARCH']
            params_prediction['model_inputs'] = params['INPUTS_IDS_MODEL']
            params_prediction['model_outputs'] = params['OUTPUTS_IDS_MODEL']
            params_prediction['dataset_inputs'] = params['INPUTS_IDS_DATASET']
            params_prediction['dataset_outputs'] = params['OUTPUTS_IDS_DATASET']
            params_prediction['normalize_probs'] = params['NORMALIZE_SAMPLING']
            params_prediction['alpha_factor'] = params['ALPHA_FACTOR']
            params_prediction['pos_unk'] = params['POS_UNK']
            mapping = None if dataset.mapping == dict() else dataset.mapping
            if params['POS_UNK']:
                params_prediction['heuristic'] = params['HEURISTIC']
                input_text_id = params['INPUTS_IDS_DATASET'][0]
                vocab_src = dataset.vocabulary[input_text_id]['idx2words']
            else:
                input_text_id = None
                vocab_src = None
                mapping = None
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
                sources = map(lambda x: x.strip(), open(args.text, 'r').read().split('\n'))
                sources = sources[:-1] if len(sources[-1]) == 0 else sources
                heuristic = params_prediction['heuristic']
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
                if params_prediction['pos_unk']:
                    sources = map(lambda x: x.strip().split(), open(args.text, 'r').read().split('\n'))
                    heuristic = params_prediction['heuristic']
                else:
                    alphas = None
                    heuristic = None
                    sources = None
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
            if params['SAMPLING_SAVE_MODE'] == 'list':
                list2file(filepath, predictions)
                if args.n_best:
                    nbest2file(filepath + '.nbest', n_best_predictions)
            else:
                raise Exception('Only "list" is allowed in "SAMPLING_SAVE_MODE"')
