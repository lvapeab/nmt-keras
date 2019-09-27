# This predict script doesn't work yet -- drj 2019-09

import argparse
import ast
import os
import re
import sys

import config
from data_engine.prepare_data import build_dataset, update_dataset_from_file
from keras_wrapper.cnn_model import loadModel, updateModel
from keras_wrapper.dataset import loadDataset, saveDataset
from keras_wrapper.extra.callbacks import *
from nmt_keras.model_zoo import TranslationModel
from utils.utils import update_parameters

from numpy.random import seed

logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser("Train or sample NMT models")
    parser.add_argument("--dataset", required=True,
        help="dataset file (.pkl) to use")
    parser.add_argument("--model", required=True,
        help="model file (.h5) to use")
    parser.add_argument("-c", "--config", required=False, help="Config pkl for loading the model configuration. "
                                                               "If not specified, hyperparameters "
                                                               "are read from config.py")
    parser.add_argument("changes", nargs="*", help="Changes to config. "
                                                   "Following the syntax Key=Value",
                        default="")
    return parser.parse_args()


def apply_NMT_model(params, args):
    """
    Sample from a previously trained model.

    :param params: Dictionary of network hyperparameters.
    :return: None
    """

    params['PRED_VOCAB'] = args.dataset
    dataset_voc = loadDataset(params['PRED_VOCAB'])
    dataset = build_dataset(params, dataset_voc.vocabulary, dataset_voc.vocabulary_len)

    params['INPUT_VOCABULARY_SIZE'] = dataset.vocabulary_len[params['INPUTS_IDS_DATASET'][0]]
    params['OUTPUT_VOCABULARY_SIZE'] = dataset.vocabulary_len['target_text']

    # Load model

    nmt_model = loadModel(args.model, -1, full_path=True)

    """
    nmt_model = TranslationModel(params,
                                     model_type=params['MODEL_TYPE'],
                                     verbose=params['VERBOSE'],
                                     model_name=params['MODEL_NAME'],
                                     set_optimizer=False,
                                     vocabularies=dataset.vocabulary,
                                     store_path=params['STORE_PATH'],
                                     trainable_pred=True, trainable_est=True,
                                     weights_path=None)
    nmt_model = updateModel(nmt_model, params['STORE_PATH'], params['RELOAD'], reload_epoch=params['RELOAD_EPOCH'])
    nmt_model.setParams(params)
    nmt_model.setOptimizer()
    """

    inputMapping = dict()
    for i, id_in in enumerate(params['INPUTS_IDS_DATASET']):
        pos_source = dataset.ids_inputs.index(id_in)
        id_dest = nmt_model.ids_inputs[i]
        inputMapping[id_dest] = pos_source
    nmt_model.setInputsMapping(inputMapping)

    outputMapping = dict()
    for i, id_out in enumerate(params['OUTPUTS_IDS_DATASET']):
        pos_target = dataset.ids_outputs.index(id_out)
        id_dest = nmt_model.ids_outputs[i]
        outputMapping[id_dest] = pos_target
    nmt_model.setOutputsMapping(outputMapping)
    nmt_model.setOptimizer()

    for s in params["EVAL_ON_SETS"]:
        # Evaluate training
        extra_vars = {'language': params.get('TRG_LAN', 'en'),
                      'n_parallel_loaders': params['PARALLEL_LOADERS'],
                      'tokenize_f': eval('dataset.' + params['TOKENIZATION_METHOD']),
                      'detokenize_f': eval('dataset.' + params['DETOKENIZATION_METHOD']),
                      'apply_detokenization': params['APPLY_DETOKENIZATION'],
                      'tokenize_hypotheses': params['TOKENIZE_HYPOTHESES'],
                      'tokenize_references': params['TOKENIZE_REFERENCES']}
        extra_vars[s] = dict()
        # True when we should score against a reference
        reference_score = not params.get('NO_REF', False)

        extra_vars[s]['reference_score'] = reference_score
        if reference_score: 
            extra_vars[s]['references'] = dataset.extra_variables[s][params['OUTPUTS_IDS_DATASET'][0]]

        input_text_id = params['INPUTS_IDS_DATASET'][0]
        vocab_x = dataset.vocabulary[input_text_id]['idx2words']
        vocab_y = dataset.vocabulary[params['INPUTS_IDS_DATASET'][1]]['idx2words']

        callback_metric = PrintPerformanceMetricOnEpochEndOrEachNUpdates(nmt_model,
                                                                         dataset,
                                                                         gt_id=params['OUTPUTS_IDS_DATASET'][0],
                                                                         metric_name=params['METRICS'],
                                                                         set_name=params['EVAL_ON_SETS'],
                                                                         batch_size=params['BATCH_SIZE'],
                                                                         each_n_epochs=params['EVAL_EACH'],
                                                                         extra_vars=extra_vars,
                                                                         reload_epoch=params['RELOAD'],
                                                                         is_text=True,
                                                                         input_text_id=input_text_id,
                                                                         save_path=nmt_model.model_path,
                                                                         index2word_y=vocab_y,
                                                                         index2word_x=vocab_x,
                                                                         sampling_type=params['SAMPLING'],
                                                                         beam_search=params['BEAM_SEARCH'],
                                                                         start_eval_on_epoch=params[
                                                                             'START_EVAL_ON_EPOCH'],
                                                                         write_samples=True,
                                                                         write_type=params['SAMPLING_SAVE_MODE'],
                                                                         eval_on_epochs=params['EVAL_EACH_EPOCHS'],
                                                                         save_each_evaluation=False,
                                                                         verbose=params['VERBOSE'])

        callback_metric.evaluate(params['RELOAD'], counter_name='epoch' if params['EVAL_EACH_EPOCHS'] else 'update')


def zzzbuildCallbacks(params, model, dataset):
    """
    Builds the selected set of callbacks run during the training of the model.

    :param params: Dictionary of network hyperparameters.
    :param model: Model instance on which to apply the callback.
    :param dataset: Dataset instance on which to apply the callback.
    :return:
    """

    callbacks = []

    if params['METRICS'] or params['SAMPLE_ON_SETS']:
        # Evaluate training
        extra_vars = {'language': params.get('TRG_LAN', 'en'),
                      'n_parallel_loaders': params['PARALLEL_LOADERS'],
                      'tokenize_f': eval('dataset.' + params.get('TOKENIZATION_METHOD', 'tokenize_none')),
                      'detokenize_f': eval('dataset.' + params.get('DETOKENIZATION_METHOD', 'detokenize_none')),
                      'apply_detokenization': params.get('APPLY_DETOKENIZATION', False),
                      'tokenize_hypotheses': params.get('TOKENIZE_HYPOTHESES', True),
                      'tokenize_references': params.get('TOKENIZE_REFERENCES', True)
                      }

        input_text_id = params['INPUTS_IDS_DATASET'][0]
        vocab_x = dataset.vocabulary[input_text_id]['idx2words']
        vocab_y = dataset.vocabulary[params['INPUTS_IDS_DATASET'][1]]['idx2words']
        if params['BEAM_SEARCH']:
            extra_vars['beam_size'] = params.get('BEAM_SIZE', 6)
            extra_vars['state_below_index'] = params.get('BEAM_SEARCH_COND_INPUT', -1)
            extra_vars['maxlen'] = params.get('MAX_OUTPUT_TEXT_LEN_TEST', 30)
            extra_vars['optimized_search'] = params.get('OPTIMIZED_SEARCH', True)
            extra_vars['model_inputs'] = params['INPUTS_IDS_MODEL']
            extra_vars['model_outputs'] = params['OUTPUTS_IDS_MODEL']
            extra_vars['dataset_inputs'] = params['INPUTS_IDS_DATASET']
            extra_vars['dataset_outputs'] = params['OUTPUTS_IDS_DATASET']
            extra_vars['search_pruning'] = params.get('SEARCH_PRUNING', False)
            extra_vars['normalize_probs'] = params.get('NORMALIZE_SAMPLING', False)
            extra_vars['alpha_factor'] = params.get('ALPHA_FACTOR', 1.)
            extra_vars['coverage_penalty'] = params.get('COVERAGE_PENALTY', False)
            extra_vars['length_penalty'] = params.get('LENGTH_PENALTY', False)
            extra_vars['length_norm_factor'] = params.get('LENGTH_NORM_FACTOR', 0.0)
            extra_vars['coverage_norm_factor'] = params.get('COVERAGE_NORM_FACTOR', 0.0)
            extra_vars['pos_unk'] = params['POS_UNK']
            extra_vars['output_max_length_depending_on_x'] = params.get('MAXLEN_GIVEN_X', True)
            extra_vars['output_max_length_depending_on_x_factor'] = params.get('MAXLEN_GIVEN_X_FACTOR', 3)
            extra_vars['output_min_length_depending_on_x'] = params.get('MINLEN_GIVEN_X', True)
            extra_vars['output_min_length_depending_on_x_factor'] = params.get('MINLEN_GIVEN_X_FACTOR', 2)

            if params['POS_UNK']:
                extra_vars['heuristic'] = params['HEURISTIC']
                if params['HEURISTIC'] > 0:
                    extra_vars['mapping'] = dataset.mapping

        if params['METRICS']:
            for s in params['EVAL_ON_SETS']:
                extra_vars[s] = dict()
                extra_vars[s]['references'] = dataset.extra_variables[s][params['OUTPUTS_IDS_DATASET'][0]]
            callback_metric = PrintPerformanceMetricOnEpochEndOrEachNUpdates(model,
                                                                             dataset,
                                                                             gt_id=[params['OUTPUTS_IDS_DATASET'][0]],
                                                                             metric_name=params['METRICS'],
                                                                             set_name=params['EVAL_ON_SETS'],
                                                                             batch_size=params['BATCH_SIZE'],
                                                                             each_n_epochs=params['EVAL_EACH'],
                                                                             extra_vars=extra_vars,
                                                                             reload_epoch=params['RELOAD'],
                                                                             is_text=True,
                                                                             input_text_id=input_text_id,
                                                                             index2word_y=vocab_y,
                                                                             # index2word_y=dataset.vocabulary[params['OUTPUTS_IDS_DATASET'][0]]['idx2words'],
                                                                             index2word_x=vocab_x,
                                                                             sampling_type=params['SAMPLING'],
                                                                             beam_search=params['BEAM_SEARCH'],
                                                                             save_path=model.model_path,
                                                                             start_eval_on_epoch=params[
                                                                                 'START_EVAL_ON_EPOCH'],
                                                                             write_samples=True,
                                                                             write_type=params['SAMPLING_SAVE_MODE'],
                                                                             eval_on_epochs=params['EVAL_EACH_EPOCHS'],
                                                                             save_each_evaluation=params[
                                                                                 'SAVE_EACH_EVALUATION'],
                                                                             verbose=params['VERBOSE'],)

            callbacks.append(callback_metric)

        # if params['SAMPLE_ON_SETS']:
        #     callback_sampling = SampleEachNUpdates(model,
        #                                            dataset,
        #                                            gt_id=params['OUTPUTS_IDS_DATASET'][0],
        #                                            set_name=params['SAMPLE_ON_SETS'],
        #                                            n_samples=params['N_SAMPLES'],
        #                                            each_n_updates=params['SAMPLE_EACH_UPDATES'],
        #                                            extra_vars=extra_vars,
        #                                            reload_epoch=params['RELOAD'],
        #                                            batch_size=params['BATCH_SIZE'],
        #                                            is_text=True,
        #                                            index2word_x=vocab_x,
        #                                            index2word_y=vocab_y,
        #                                            print_sources=True,
        #                                            in_pred_idx=params['INPUTS_IDS_DATASET'][0],
        #                                            sampling_type=params['SAMPLING'],  # text info
        #                                            beam_search=params['BEAM_SEARCH'],
        #                                            start_sampling_on_epoch=params['START_SAMPLING_ON_EPOCH'],
        #                                            verbose=params['VERBOSE'])
        #     callbacks.append(callback_sampling)
    return callbacks


def check_params(params):
    """
    Checks some typical parameters and warns if something wrong was specified.
    :param params: Model instance on which to apply the callback.
    :return: None
    """
    if params['POS_UNK']:
        assert params['OPTIMIZED_SEARCH'], 'Unknown words replacement requires ' \
                                           'to use the optimized search ("OPTIMIZED_SEARCH" parameter).'
    if params['COVERAGE_PENALTY']:
        assert params['OPTIMIZED_SEARCH'], 'The application of "COVERAGE_PENALTY" requires ' \
                                           'to use the optimized search ("OPTIMIZED_SEARCH" parameter).'
    if params['SRC_PRETRAINED_VECTORS'] and params['SRC_PRETRAINED_VECTORS'][:-1] != '.npy':
        warnings.warn('It seems that the pretrained word vectors provided for the target text are not in npy format.'
                      'You should preprocess the word embeddings with the "utils/preprocess_*_word_vectors.py script.')

    if params['TRG_PRETRAINED_VECTORS'] and params['TRG_PRETRAINED_VECTORS'][:-1] != '.npy':
        warnings.warn('It seems that the pretrained word vectors provided for the target text are not in npy format.'
                      'You should preprocess the word embeddings with the "utils/preprocess_*_word_vectors.py script.')


if __name__ == "__main__":
    args = parse_args()
    print(args)
    parameters = config.load_parameters()
    if args.config is not None:
        parameters = update_parameters(parameters, pkl2dict(args.config))
    try:
        for arg in args.changes:
            try:
                k, v = arg.split('=')
            except ValueError:
                print('Overwriting arguments must have the form key=value.\n This one is: %s' % str(args.changes))
                exit(1)
            if '_' in v:
                parameters[k] = v
            else:
                try:
                    parameters[k] = ast.literal_eval(v)
                except ValueError:
                    parameters[k] = v
    except ValueError:
        print("Error processing arguments: {!r}".format(arg))
        exit(2)

    rnd_seed = parameters.get('RND_SEED', None)
    if rnd_seed != None:
        seed(rnd_seed)
    
    logging.info('Running sampling.')
    parameters['DATA_ROOT_PATH'] = "examples/wmt15/"
    parameters["SRC_LAN"] = "source"
    parameters["TRG_LAN"] = "target"
    # parameters["MODEL_NAME"] =  "trained_models/wmt15_sourcetarget_EncSent/epoch_2"

    # NMT Keras expects model path to appear without the .h5
    if args.model.endswith(".h5"):
        args.model = args.model[:-3]
    # Directory containing model
    model_dir, file_name = os.path.split(args.model)
    _, parameters["MODEL_NAME"] = os.path.split(model_dir)
    parameters["STORE_PATH"] = "trained_models/" + parameters["MODEL_NAME"]
    del parameters["TASK_NAME"]
    assert file_name.startswith("epoch_")
    parameters["RELOAD"] = file_name.replace("epoch_", "")

    from nmt_keras import model_zoo
    from keras.utils import CustomObjectScope
    with CustomObjectScope(vars(model_zoo)):
        apply_NMT_model(parameters, args)
