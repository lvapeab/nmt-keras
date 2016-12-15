import logging
import argparse

from config import load_parameters
from keras_wrapper.dataset import loadDataset
from keras_wrapper.cnn_model import loadModel
from keras_wrapper.beam_search_ensemble import BeamSearchEnsemble

import utils

logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser("Apply several translation models for making predictions")
    parser.add_argument("-ds", "--dataset", required=True, help="Dataset instance with data")
    parser.add_argument("-s", "--splits",  nargs='+', required=False, default='val', help="Splits to sample. "
                                                                                         "Should be already included"
                                                                                         "into the dataset object.")
    parser.add_argument("-d", "--dest",  required=False, help="File to save translations in")
    parser.add_argument("--not-eval", action='store_true', default=False, help="Do not compute metrics for the output")
    parser.add_argument("-e", "--eval-output", required=False, help="Write evaluation results to file")
    parser.add_argument("-v", "--verbose", required=False,  action='store_true', default=False, help="Be verbose")
    parser.add_argument("-c", "--config",  required=False, help="Config pkl for loading the model configuration. "
                                                                "If not specified, hyperparameters are read from config.py")
    parser.add_argument("--models", nargs='+', required=True, help="path to the models")
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()
    models = args.models
    print "Using an ensemble of %d models" % len(args.models)
    models = [loadModel(m, -1, full_path=True) for m in args.models]
    dataset = loadDataset(args.dataset)
    if args.config is None:
        params = load_parameters()
    else:
        params = utils.read_write.pkl2dict(args.config)
    params['INPUT_VOCABULARY_SIZE'] = dataset.vocabulary_len[params['INPUTS_IDS_DATASET'][0]]
    params['OUTPUT_VOCABULARY_SIZE'] = dataset.vocabulary_len[params['OUTPUTS_IDS_DATASET'][0]]
    # Apply sampling
    extra_vars = dict()
    extra_vars['tokenize_f'] = eval('dataset.' + params['TOKENIZATION_METHOD'])
    for s in args.splits:
        # Apply model predictions
        params_prediction = {'batch_size': params['BATCH_SIZE'],
                             'n_parallel_loaders': params['PARALLEL_LOADERS'],
                             'predict_on_sets': [s]}

        # Convert predictions into sentences
        vocab = dataset.vocabulary[params['OUTPUTS_IDS_DATASET'][0]]['idx2words']

        if params['BEAM_SEARCH']:
            params_prediction['beam_size'] = params['BEAM_SIZE']
            params_prediction['maxlen'] = params['MAX_OUTPUT_TEXT_LEN']
            params_prediction['model_inputs'] = params['INPUTS_IDS_MODEL']
            params_prediction['model_outputs'] = params['OUTPUTS_IDS_MODEL']
            params_prediction['dataset_inputs'] = params['INPUTS_IDS_DATASET']
            params_prediction['dataset_outputs'] = params['OUTPUTS_IDS_DATASET']
            params_prediction['normalize'] = params['NORMALIZE_SAMPLING']
            params_prediction['alpha_factor'] = params['ALPHA_FACTOR']

            beam_searcher = BeamSearchEnsemble(models, dataset, params_prediction, verbose=args.verbose)
            predictions = beam_searcher.BeamSearchNet()[s]
            predictions = models[0].decode_predictions_beam_search(predictions,
                                                                   vocab,
                                                                   verbose=params['VERBOSE'])
        # Store result
        if args.dest is not None:
            filepath = args.dest  # results file
            if params['SAMPLING_SAVE_MODE'] == 'list':
                utils.read_write.list2file(filepath, predictions)
            else:
                raise Exception, 'Only "list" is allowed in "SAMPLING_SAVE_MODE"'
        if args.not_eval is False:
            # Evaluate if any metric in params['METRICS']
            for metric in params['METRICS']:
                logging.info('Evaluating on metric ' + metric)
                # Evaluate on the chosen metric
                extra_vars[s] = dict()
                extra_vars[s]['references'] = dataset.extra_variables[s][params['OUTPUTS_IDS_DATASET'][0]]
                metrics = utils.evaluation.select[metric](
                    pred_list=predictions,
                    verbose=1,
                    extra_vars=extra_vars,
                    split=s)
                if args.eval_output is not None:
                    filepath = args.eval_output  # results file
                    # Print results to file
                    with open(filepath, 'w') as f:
                        header = ''
                        line = ''
                        for metric_ in sorted(metrics):
                            value = metrics[metric_]
                            header += metric_ + ','
                            line += str(value) + ','
                        f.write(header + '\n')
                        f.write(line + '\n')
                    logging.info('Done evaluating on metric ' + metric)
