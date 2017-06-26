import logging
import argparse
from data_engine.prepare_data import update_dataset_from_file
from config import load_parameters
from keras_wrapper.dataset import loadDataset
from keras_wrapper.cnn_model import loadModel
from keras_wrapper.model_ensemble import BeamSearchEnsemble
from keras_wrapper.extra.read_write import pkl2dict, list2file, numpy2file

logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser("Use several translation models for scoring source--target pairs")
    parser.add_argument("-ds", "--dataset", required=True, help="Dataset instance with data")
    parser.add_argument("-src", "--source", required=False, help="Text file with source sentences")
    parser.add_argument("-trg", "--target", required=False, help="Text file with target sentences")
    parser.add_argument("-s", "--splits", nargs='+', required=False, default=['val'], help="Splits to sample. "
                                                                                           "Should be already included"
                                                                                           "into the dataset object.")
    parser.add_argument("-d", "--dest", required=False, help="File to save scores in")
    parser.add_argument("-v", "--verbose", required=False, action='store_true', default=False, help="Be verbose")
    parser.add_argument("-c", "--config", required=False, help="Config pkl for loading the model configuration. "
                                                               "If not specified, hyperparameters "
                                                               "are read from config.py")
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
    if args.source is not None:
        dataset = update_dataset_from_file(dataset, args.source, params, splits=args.splits,
                                           output_text_filename=args.target, compute_state_below=True)

    params['INPUT_VOCABULARY_SIZE'] = dataset.vocabulary_len[params['INPUTS_IDS_DATASET'][0]]
    params['OUTPUT_VOCABULARY_SIZE'] = dataset.vocabulary_len[params['OUTPUTS_IDS_DATASET'][0]]
    # Apply scoring
    extra_vars = dict()
    extra_vars['tokenize_f'] = eval('dataset.' + params['TOKENIZATION_METHOD'])
    for s in args.splits:
        # Apply model predictions
        params_prediction = {'max_batch_size': params['BATCH_SIZE'],
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
            params_prediction['normalize_probs'] = params.get('NORMALIZE_SAMPLING', False)
            params_prediction['alpha_factor'] = params.get('ALPHA_FACTOR', 1.0)
            params_prediction['coverage_penalty'] = params.get('COVERAGE_PENALTY', False)
            params_prediction['length_penalty'] = params.get('LENGTH_PENALTY', False)
            params_prediction['length_norm_factor'] = params.get('LENGTH_NORM_FACTOR', 0.0)
            params_prediction['coverage_norm_factor'] = params.get('COVERAGE_NORM_FACTOR', 0.0)
            params_prediction['pos_unk'] = params.get('POS_UNK', False)
            params_prediction['output_max_length_depending_on_x'] = params.get('MAXLEN_GIVEN_X', True)
            params_prediction['output_max_length_depending_on_x_factor'] = params.get('MAXLEN_GIVEN_X_FACTOR', 3)
            params_prediction['output_min_length_depending_on_x'] = params.get('MINLEN_GIVEN_X', True)
            params_prediction['output_min_length_depending_on_x_factor'] = params.get('MINLEN_GIVEN_X_FACTOR', 2)

            mapping = None if dataset.mapping == dict() else dataset.mapping
            if params['POS_UNK']:
                params_prediction['heuristic'] = params['HEURISTIC']
                input_text_id = params['INPUTS_IDS_DATASET'][0]
                vocab_src = dataset.vocabulary[input_text_id]['idx2words']
            else:
                input_text_id = None
                vocab_src = None
                mapping = None
            beam_searcher = BeamSearchEnsemble(models, dataset, params_prediction, verbose=args.verbose)
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
            print scores
