import sys
import ast
import argparse
import copy
import time
from timeit import default_timer as timer

from keras_wrapper.beam_search_interactive import BeamSearchEnsemble
from utils.utils import *
from config import load_parameters
from config_online import load_parameters as load_parameters_online
from data_engine.prepare_data import build_dataset, update_dataset_from_file
from model_zoo import TranslationModel
from keras.layers import Input, Lambda, RemoveMask
from keras.models import Model
from keras import backend as K
from keras.optimizers import PAS, PPAS
from keras.objectives import log_diff
from keras_wrapper.cnn_model import loadModel, saveModel, updateModel
from keras_wrapper.dataset import loadDataset
from keras_wrapper.online_trainer import OnlineTrainer
from keras_wrapper.extra.callbacks import *

logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser("Train or sample NMT models")
    parser.add_argument("-c", "--config", required=False, help="Config pkl for loading the model configuration. "
                                                               "If not specified, hyperparameters "
                                                               "are read from config.py")
    parser.add_argument("-o", "--online",
                        action='store_true', default=False, required=False, help="Online training mode. ")
    parser.add_argument("-s", "--splits",  nargs='+', required=False, default=['val'],
                        help="Splits to train on. Should be already included into the dataset object.")
    parser.add_argument("-ds", "--dataset", required=False, help="Dataset instance with data")
    parser.add_argument("-m", "--models", nargs='*', required=False, help="Models to load", default="")
    parser.add_argument("-src", "--source", help="File of source hypothesis", required=False)
    parser.add_argument("-trg", "--references", help="Reference sentence", required=False)
    parser.add_argument("-hyp", "--hypotheses", required=False, help="Store hypothesis to this file")
    parser.add_argument("-v", "--verbose", required=False, default=0, type=int, help="Verbosity level")
    parser.add_argument("-ch", "--changes",  nargs="*", help="Changes to config, following the syntax Key=Value",
                        default="")

    return parser.parse_args()


def train_model(params):
    """
    Training function. Sets the training parameters from params. Build or loads the model and launches the training.
    :param params: Dictionary of network hyperparameters.
    :return: None
    """

    if params['RELOAD'] > 0:
        logging.info('Resuming training.')

    check_params(params)

    # Load data
    dataset = build_dataset(params)
    params['INPUT_VOCABULARY_SIZE'] = dataset.vocabulary_len[params['INPUTS_IDS_DATASET'][0]]
    params['OUTPUT_VOCABULARY_SIZE'] = dataset.vocabulary_len[params['OUTPUTS_IDS_DATASET'][0]]

    # Build model
    if params['RELOAD'] == 0:  # build new model
        nmt_model = TranslationModel(params, type=params['MODEL_TYPE'], verbose=params['VERBOSE'],
                                     model_name=params['MODEL_NAME'], vocabularies=dataset.vocabulary,
                                     store_path=params['STORE_PATH'])
        dict2pkl(params, params['STORE_PATH'] + '/config')

        # Define the inputs and outputs mapping from our Dataset instance to our model
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

    else:  # resume from previously trained model
        nmt_model = TranslationModel(params,
                                     type=params['MODEL_TYPE'],
                                     verbose=params['VERBOSE'],
                                     model_name=params['MODEL_NAME'],
                                     vocabularies=dataset.vocabulary,
                                     store_path=params['STORE_PATH'],
                                     set_optimizer=False,
                                     clear_dirs=False)
        nmt_model = updateModel(nmt_model, params['STORE_PATH'], params['RELOAD'])
        nmt_model.setParams(params)
        nmt_model.setOptimizer()

    # Callbacks
    callbacks = buildCallbacks(params, nmt_model, dataset)

    # Training
    total_start_time = timer()

    logger.debug('Starting training!')
    training_params = {'n_epochs': params['MAX_EPOCH'],
                       'batch_size': params['BATCH_SIZE'],
                       'homogeneous_batches': params['HOMOGENEOUS_BATCHES'], 'maxlen': params['MAX_OUTPUT_TEXT_LEN'],
                       'joint_batches': params['JOINT_BATCHES'],
                       'lr_decay': params['LR_DECAY'], 'lr_gamma': params['LR_GAMMA'],
                       'epochs_for_save': params['EPOCHS_FOR_SAVE'], 'verbose': params['VERBOSE'],
                       'eval_on_sets': params['EVAL_ON_SETS_KERAS'], 'n_parallel_loaders': params['PARALLEL_LOADERS'],
                       'extra_callbacks': callbacks, 'reload_epoch': params['RELOAD'], 'epoch_offset': params['RELOAD'],
                       'data_augmentation': params['DATA_AUGMENTATION'],
                       'patience': params.get('PATIENCE', 0),# early stopping parameters
                       'metric_check': params.get('STOP_METRIC', None),
                       'eval_on_epochs': params.get('EVAL_EACH_EPOCHS', True),
                       'each_n_epochs': params.get('EVAL_EACH', 1),
                       'start_eval_on_epoch': params.get('START_EVAL_ON_EPOCH', 0)}
    nmt_model.trainNet(dataset, training_params)

    total_end_time = timer()
    time_difference = total_end_time - total_start_time
    logging.info('In total is {0:.2f}s = {1:.2f}m'.format(time_difference, time_difference / 60.0))


def train_model_online(params, source_filename, target_filename, models_path=None, dataset=None, store_hypotheses=None, verbose=0):
    """
    Training function. Sets the training parameters from params. Build or loads the model and launches the training.
    :param params: Dictionary of network hyperparameters.
    :return: None
    """

    logging.info('Starting online training.')

    check_params(params)
    # Load data
    if dataset is None:
        dataset = build_dataset(params)
    params['INPUT_VOCABULARY_SIZE'] = dataset.vocabulary_len[params['INPUTS_IDS_DATASET'][0]]
    params['OUTPUT_VOCABULARY_SIZE'] = dataset.vocabulary_len[params['OUTPUTS_IDS_DATASET'][0]]

    # Load models
    if models_path is not None:
        logging.info('Loading models from %s'%str(models_path))
        model_instances = [TranslationModel(params,
                                            type=params['MODEL_TYPE'],
                                            verbose=params['VERBOSE'],
                                            model_name=params['MODEL_NAME'] + '_' + str(i),
                                            vocabularies=dataset.vocabulary,
                                            store_path=params['STORE_PATH'],
                                            set_optimizer=False,
                                            clear_dirs=False)
                           for i in range(len(models_path))]
        models = [updateModel(model, path, -1, full_path=True) for (model, path) in zip(model_instances, models_path)]
    else:
        raise Exception, 'Online mode requires an already trained model!'

    trainer_models = []
    if params['USE_CUSTOM_LOSS']:
        ###### ADD ADDITIONAL INPUT LAYER TO MODELS IN ORDER TO TRAIN WITH CUSTOM LOSS FUNCTION ##
        for nmt_model in models:
            hyp_in = Input(name="hyp_input", batch_shape=tuple([None, None, None]))
            yref_in = Input(name="yref_input", batch_shape=tuple([None, None, None]))

            model_out = RemoveMask()(nmt_model.model.outputs[0])
            loss_out = Lambda(log_diff, output_shape=(1,), name='custom_loss', supports_masking=False)([model_out, yref_in, hyp_in])
            trainer_model = Model(input=nmt_model.model.input + [yref_in, hyp_in], output=loss_out)

            trainer_models.append(trainer_model)

            # Set custom optimizer
            weights = trainer_model.trainable_weights
            weights.sort(key=lambda x: x.name if x.name else x.auto_name)
            weights_shapes = [K.get_variable_shape(w) for w in weights]
            subgradientOpt = eval(params['OPTIMIZER'])(weights_shapes, lr=params['LR'], c=params['C'])
            trainer_model.compile(loss={'custom_loss': lambda y_true, y_pred: y_pred}, optimizer=subgradientOpt)

            for nmt_model in models:
                nmt_model.setParams(params)
    else:
        for nmt_model in models:
            nmt_model.setParams(params)
            nmt_model.setOptimizer()

    # Apply model predictions
    params_prediction = {# Decoding params
                         #'predict_on_sets': [s],
                         'beam_size': params['BEAM_SIZE'],
                         'maxlen': params['MAX_OUTPUT_TEXT_LEN_TEST'],
                         'optimized_search': params['OPTIMIZED_SEARCH'],
                         'model_inputs': params['INPUTS_IDS_MODEL'],
                         'model_outputs': params['OUTPUTS_IDS_MODEL'],
                         'dataset_inputs': params['INPUTS_IDS_DATASET'],
                         'dataset_outputs': params['OUTPUTS_IDS_DATASET'],
                         'normalize': params['NORMALIZE_SAMPLING'],
                         'alpha_factor': params['ALPHA_FACTOR'],
                         'pos_unk': params['POS_UNK'],
                         'state_below_index': -1,
                         'output_text_index': 0,
    }
    params_training = {  #Traning params
                         'n_epochs': params['MAX_EPOCH'],
                         'shuffle': False,
                         'use_custom_loss': params['USE_CUSTOM_LOSS'],
                         'batch_size': params['BATCH_SIZE'],
                         'homogeneous_batches': False,
                         'lr_decay': params['LR_DECAY'],
                         'lr_gamma': params['LR_GAMMA'],
                         'epochs_for_save': -1,
                         'verbose': verbose,
                         'eval_on_sets': params['EVAL_ON_SETS_KERAS'],
                         'n_parallel_loaders': params['PARALLEL_LOADERS'],
                         'extra_callbacks': [], #callbacks,
                         'reload_epoch': params['RELOAD'],
                         'epoch_offset': params['RELOAD'],
                         'data_augmentation': params['DATA_AUGMENTATION'],
                         'patience': params.get('PATIENCE', 0),
                         'metric_check': params.get('STOP_METRIC', None),
                         'eval_on_epochs': params.get('EVAL_EACH_EPOCHS', True),
                         'each_n_epochs': params.get('EVAL_EACH', 1),
                         'start_eval_on_epoch': params.get('START_EVAL_ON_EPOCH', 0)
                         }

    # Create sampler
    logging.info('Creating sampler...')
    beam_searcher = BeamSearchEnsemble(models, dataset, params_prediction, verbose=verbose)
    params_prediction = copy.copy(params_prediction)
    params_prediction['store_hypotheses'] = store_hypotheses

    # Create trainer
    logging.info('Creating trainer...')
    if params["USE_CUSTOM_LOSS"]:
        online_trainer = OnlineTrainer(trainer_models,
                                       dataset,
                                       beam_searcher,
                                       params_prediction,
                                       params_training,
                                       verbose=verbose)
    else:
        online_trainer = OnlineTrainer(models,
                                       dataset,
                                       beam_searcher,
                                       params_prediction,
                                       params_training,
                                       verbose=verbose)

    # Open new data
    ftrg = open(target_filename, 'r')
    target_lines = ftrg.read().split('\n')[:-1]
    ftrg.close()
    fsrc = open(source_filename, 'r')
    source_lines = fsrc.read().split('\n')[:-1]
    fsrc.close()
    source_lines = source_lines[:-1] if source_lines[-1] == '' else source_lines
    target_lines = target_lines[:-1] if target_lines[-1] == '' else target_lines
    n_lines = len(source_lines)
    assert len(source_lines) == len(target_lines), 'Number of source and target lines must match'
    # Empty dest file
    if store_hypotheses:
        logging.info('Storing htypotheses in: %s' % store_hypotheses)
        open(store_hypotheses, 'w').close()

    start_time = time.time()
    eta = -1
    for n_line, (source_line, target_line) in enumerate(zip(source_lines, target_lines)):

        src_seq = dataset.loadText([source_line],
                                   dataset.vocabulary[params['INPUTS_IDS_DATASET'][0]],
                                   params['MAX_OUTPUT_TEXT_LEN_TEST'],
                                   0,
                                   fill=dataset.fill_text[params['INPUTS_IDS_DATASET'][0]],
                                   pad_on_batch=dataset.pad_on_batch[params['INPUTS_IDS_DATASET'][0]],
                                   words_so_far=False,
                                   loading_X=True)[0]
        if verbose > 1:
            logging.info('Input sentence:  %s' % str(src_seq))
            logging.info('Parsed sentence: %s' % str(map(lambda x: dataset.vocabulary[params['INPUTS_IDS_DATASET'][0]]['idx2words'][x], src_seq[0])))
        state_below = dataset.loadText([target_line],
                                   dataset.vocabulary[params['OUTPUTS_IDS_DATASET'][0]],
                                   params['MAX_OUTPUT_TEXT_LEN_TEST'],
                                   1,
                                   fill=dataset.fill_text[params['INPUTS_IDS_DATASET'][-1]],
                                   pad_on_batch=dataset.pad_on_batch[params['INPUTS_IDS_DATASET'][-1]],
                                   words_so_far=False,
                                   loading_X=True)[0]
        trg_seq = dataset.loadTextOneHot([target_line],
                                         vocabularies=dataset.vocabulary[params['OUTPUTS_IDS_DATASET'][0]],
                                         vocabulary_len=dataset.vocabulary_len[params['OUTPUTS_IDS_DATASET'][0]],
                                         max_len=params['MAX_OUTPUT_TEXT_LEN_TEST'],
                                         offset=0,
                                         fill=dataset.fill_text[params['OUTPUTS_IDS_DATASET'][0]],
                                         pad_on_batch=dataset.pad_on_batch[params['OUTPUTS_IDS_DATASET'][0]],
                                         words_so_far=False,
                                         sample_weights=params['SAMPLE_WEIGHTS'],
                                         loading_X=False)
        if verbose > 1:
            logging.info('Output sentence:  %s' % str(target_line))
            logging.info('Parsed sentence (state below): %s ' % map(lambda x: dataset.vocabulary[params['OUTPUTS_IDS_DATASET'][0]]['idx2words'][x], state_below[0]))

        online_trainer.sample_and_train_online([src_seq, state_below], trg_seq, src_words=[source_line])
        sys.stdout.write('\r')
        sys.stdout.write("Processed %d/%d  -  ETA: %ds " % ((n_line + 1) , n_lines, int(eta)))
        sys.stdout.flush()
        eta = (n_lines - (n_line + 1)) * (time.time() - start_time) / (n_line + 1)


    sys.stdout.write('The online training took: %f secs (Speed: %f sec/sample)\n' % ((time.time() - start_time), (
                    time.time() - start_time) / n_lines))
    saveModel(nmt_model, -1,  path=params.get('STORE_PATH', 'retrained_model'), full_path=True)


def apply_NMT_model(params):
    """
    Sample from a previously trained model.

    :param params: Dictionary of network hyperparameters.
    :return: None
    """

    # Load data
    dataset = build_dataset(params)
    params['INPUT_VOCABULARY_SIZE'] = dataset.vocabulary_len[params['INPUTS_IDS_DATASET'][0]]
    params['OUTPUT_VOCABULARY_SIZE'] = dataset.vocabulary_len[params['OUTPUTS_IDS_DATASET'][0]]

    # Load model
    nmt_model = loadModel(params['STORE_PATH'], params['RELOAD'])
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
        vocab = dataset.vocabulary[params['OUTPUTS_IDS_DATASET'][0]]['idx2words']
        extra_vars[s] = dict()
        extra_vars[s]['references'] = dataset.extra_variables[s][params['OUTPUTS_IDS_DATASET'][0]]
        input_text_id = None
        vocab_src = None
        if params['BEAM_SIZE']:
            extra_vars['beam_size'] = params.get('BEAM_SIZE', 6)
            extra_vars['state_below_index'] = params.get('BEAM_SEARCH_COND_INPUT', -1)
            extra_vars['maxlen'] = params.get('MAX_OUTPUT_TEXT_LEN_TEST', 30)
            extra_vars['optimized_search'] = params.get('OPTIMIZED_SEARCH', True)
            extra_vars['model_inputs'] = params['INPUTS_IDS_MODEL']
            extra_vars['model_outputs'] = params['OUTPUTS_IDS_MODEL']
            extra_vars['dataset_inputs'] = params['INPUTS_IDS_DATASET']
            extra_vars['dataset_outputs'] = params['OUTPUTS_IDS_DATASET']
            extra_vars['normalize'] = params.get('NORMALIZE_SAMPLING', False)
            extra_vars['alpha_factor'] = params.get('ALPHA_FACTOR', 1.)
            extra_vars['pos_unk'] = params['POS_UNK']
            if params['POS_UNK']:
                extra_vars['heuristic'] = params['HEURISTIC']
                input_text_id = params['INPUTS_IDS_DATASET'][0]
                vocab_src = dataset.vocabulary[input_text_id]['idx2words']
                if params['HEURISTIC'] > 0:
                    extra_vars['mapping'] = dataset.mapping

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
                                                                         index2word_y=vocab,
                                                                         index2word_x=vocab_src,
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


def buildCallbacks(params, model, dataset):
    """
    Builds the selected set of callbacks run during the training of the model.

    :param params: Dictionary of network hyperparameters.
    :param model: Model instance on which to apply the callback.
    :param dataset: Dataset instance on which to apply the callback.
    :return:
    """

    callbacks = []

    if params['METRICS']:
        # Evaluate training
        extra_vars = {'language': params.get('TRG_LAN', 'en'),
                      'n_parallel_loaders': params['PARALLEL_LOADERS'],
                      'tokenize_f': eval('dataset.' + params.get('TOKENIZATION_METHOD', 'tokenize_none')),
                      'detokenize_f': eval('dataset.' + params.get('DETOKENIZATION_METHOD', 'detokenize_none')),
                      'apply_detokenization': params.get('APPLY_DETOKENIZATION', False),
                      'tokenize_hypotheses': params.get('TOKENIZE_HYPOTHESES', True),
                      'tokenize_references': params.get('TOKENIZE_REFERENCES', True)}

        vocab = dataset.vocabulary[params['OUTPUTS_IDS_DATASET'][0]]['idx2words']
        for s in params['EVAL_ON_SETS']:
            extra_vars[s] = dict()
            extra_vars[s]['references'] = dataset.extra_variables[s][params['OUTPUTS_IDS_DATASET'][0]]
        input_text_id = None
        vocab_src = None
        if params['BEAM_SIZE']:
            extra_vars['beam_size'] = params.get('BEAM_SIZE', 6)
            extra_vars['state_below_index'] = params.get('BEAM_SEARCH_COND_INPUT', -1)
            extra_vars['maxlen'] = params.get('MAX_OUTPUT_TEXT_LEN_TEST', 30)
            extra_vars['optimized_search'] = params.get('OPTIMIZED_SEARCH', True)
            extra_vars['model_inputs'] = params['INPUTS_IDS_MODEL']
            extra_vars['model_outputs'] = params['OUTPUTS_IDS_MODEL']
            extra_vars['dataset_inputs'] = params['INPUTS_IDS_DATASET']
            extra_vars['dataset_outputs'] = params['OUTPUTS_IDS_DATASET']
            extra_vars['normalize'] = params.get('NORMALIZE_SAMPLING', False)
            extra_vars['alpha_factor'] = params.get('ALPHA_FACTOR', 1.)
            extra_vars['pos_unk'] = params['POS_UNK']
            if params['POS_UNK']:
                extra_vars['heuristic'] = params['HEURISTIC']
                input_text_id = params['INPUTS_IDS_DATASET'][0]
                vocab_src = dataset.vocabulary[input_text_id]['idx2words']
                if params['HEURISTIC'] > 0:
                    extra_vars['mapping'] = dataset.mapping

        callback_metric = PrintPerformanceMetricOnEpochEndOrEachNUpdates(model,
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
                                                                         index2word_y=vocab,
                                                                         index2word_x=vocab_src,
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
                                                                         verbose=params['VERBOSE'])

        callbacks.append(callback_metric)

    if params['SAMPLE_ON_SETS']:
        # Write some samples
        extra_vars = {'language': params['TRG_LAN'], 'n_parallel_loaders': params['PARALLEL_LOADERS']}
        vocab_x = dataset.vocabulary[params['INPUTS_IDS_DATASET'][0]]['idx2words']
        vocab_y = dataset.vocabulary[params['OUTPUTS_IDS_DATASET'][0]]['idx2words']
        for s in params['EVAL_ON_SETS']:
            extra_vars[s] = dict()
            extra_vars[s]['references'] = dataset.extra_variables[s][params['OUTPUTS_IDS_DATASET'][0]]
            extra_vars[s]['tokenize_f'] = eval('dataset.' + params.get('TOKENIZATION_METHOD', 'tokenize_none'))
            extra_vars[s]['detokenize_f'] = eval('dataset.' + params.get('DETOKENIZATION_METHOD', 'detokenize_none'))
            extra_vars[s]['apply_detokenization'] = params.get('APPLY_DETOKENIZATION', False)
            extra_vars[s]['tokenize_hypotheses'] = params.get('TOKENIZE_HYPOTHESES', True)
            extra_vars[s]['tokenize_references'] = params.get('TOKENIZE_REFERENCES', True)

        if params['BEAM_SIZE']:
            extra_vars['beam_size'] = params['BEAM_SIZE']
            extra_vars['state_below_index'] = params.get('BEAM_SEARCH_COND_INPUT', -1)
            extra_vars['maxlen'] = params['MAX_OUTPUT_TEXT_LEN_TEST']
            extra_vars['optimized_search'] = params['OPTIMIZED_SEARCH']
            extra_vars['model_inputs'] = params['INPUTS_IDS_MODEL']
            extra_vars['model_outputs'] = params['OUTPUTS_IDS_MODEL']
            extra_vars['dataset_inputs'] = params['INPUTS_IDS_DATASET']
            extra_vars['dataset_outputs'] = params['OUTPUTS_IDS_DATASET']
            extra_vars['normalize'] = params['NORMALIZE_SAMPLING']
            extra_vars['alpha_factor'] = params['ALPHA_FACTOR']
            extra_vars['pos_unk'] = params['POS_UNK']
            if params['POS_UNK']:
                extra_vars['heuristic'] = params['HEURISTIC']
                if params['HEURISTIC'] > 0:
                    extra_vars['mapping'] = dataset.mapping

        callback_sampling = SampleEachNUpdates(model,
                                               dataset,
                                               gt_id=params['OUTPUTS_IDS_DATASET'][0],
                                               set_name=params['SAMPLE_ON_SETS'],
                                               n_samples=params['N_SAMPLES'],
                                               each_n_updates=params['SAMPLE_EACH_UPDATES'],
                                               extra_vars=extra_vars,
                                               reload_epoch=params['RELOAD'],
                                               batch_size=params['BATCH_SIZE'],
                                               is_text=True,
                                               index2word_x=vocab_x,  # text info
                                               index2word_y=vocab_y,  # text info
                                               print_sources=True,
                                               in_pred_idx=params['INPUTS_IDS_DATASET'][0],
                                               sampling_type=params['SAMPLING'],  # text info
                                               beam_search=params['BEAM_SEARCH'],
                                               start_sampling_on_epoch=params['START_SAMPLING'
                                                                              '_ON_EPOCH'],
                                               verbose=params['VERBOSE'])
        callbacks.append(callback_sampling)
    return callbacks


def check_params(params):
    """
    Checks some typical parameters and warns if something wrong was specified.
    :param params:  Model instance on which to apply the callback.
    :return: None
    """
    if params['POS_UNK']:
        assert params['OPTIMIZED_SEARCH'], 'Unknown words replacement requires ' \
                                           'to use the optimized search ("OPTIMIZED_SEARCH" parameter).'
    if params['SRC_PRETRAINED_VECTORS'] and params['SRC_PRETRAINED_VECTORS'][:-1] != '.npy':
        warnings.warn('It seems that the pretrained word vectors provided for the target text are not in npy format.'
                      'You should preprocess the word embeddings with the "utils/preprocess_*_word_vectors.py script.')

    if params['TRG_PRETRAINED_VECTORS'] and params['TRG_PRETRAINED_VECTORS'][:-1] != '.npy':
        warnings.warn('It seems that the pretrained word vectors provided for the target text are not in npy format.'
                      'You should preprocess the word embeddings with the "utils/preprocess_*_word_vectors.py script.')

if __name__ == "__main__":
    args = parse_args()
    parameters = load_parameters()
    if args.config is not None:
        parameters = update_parameters(parameters, pkl2dict(args.config))

    if args.online:
        online_parameters = load_parameters_online()
        parameters = update_parameters(parameters, online_parameters)
    try:
        for arg in args.changes:
            try:
                k, v = arg.split('=')
            except ValueError:
                print 'Overwritten arguments must have the form key=Value. \n Currently are: %s' % str(args.changes)
                exit(1)
            try:
                parameters[k] = ast.literal_eval(v)
            except:
                parameters[k] = v
    except ValueError:
        print 'Error processing arguments: (', k, ",",v,")"
        exit(2)

    check_params(parameters)
    if args.online:
        dataset = loadDataset(args.dataset)
        dataset = update_dataset_from_file(dataset, args.source, parameters,
                                           output_text_filename=args.references, splits=['train'], remove_outputs=False, compute_state_below=True)
        train_model_online(parameters, args.source, args.references, models_path=args.models, dataset=dataset, store_hypotheses=args.hypotheses, verbose=args.verbose)

    elif parameters['MODE'] == 'training':
        logging.info('Running training.')
        train_model(parameters)
    elif parameters['MODE'] == 'sampling':
        logging.info('Running sampling.')
        apply_NMT_model(parameters)

    logging.info('Done!')
