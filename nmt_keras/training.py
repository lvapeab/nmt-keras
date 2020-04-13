# -*- coding: utf-8 -*-
from __future__ import print_function
from six import iteritems
from timeit import default_timer as timer
import logging

import os

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(__name__)

from data_engine.prepare_data import build_dataset, update_dataset_from_file
from keras_wrapper.cnn_model import updateModel
from keras_wrapper.dataset import loadDataset, saveDataset
from keras_wrapper.extra.read_write import dict2pkl
from nmt_keras.model_zoo import TranslationModel
from nmt_keras.build_callbacks import buildCallbacks


def train_model(params, load_dataset=None):
    """
    Training function.

    Sets the training parameters from params.

    Build or loads the model and launches the training.

    :param dict params: Dictionary of network hyperparameters.
    :param str load_dataset: Load dataset from file or build it from the parameters.
    :return: None
    """

    if params['RELOAD'] > 0:
        logger.info('Resuming training.')
        # Load data
        if load_dataset is None:
            if params['REBUILD_DATASET']:
                logger.info('Rebuilding dataset.')
                dataset = build_dataset(params)
            else:
                logger.info('Updating dataset.')
                dataset = loadDataset(
                    os.path.join(
                        params['DATASET_STORE_PATH'],
                        'Dataset_' +
                        params['DATASET_NAME'] + '_' +
                        params['SRC_LAN'] + params['TRG_LAN'] + '.pkl')
                )

                epoch_offset = 0 if dataset.len_train == 0 else int(
                    params['RELOAD'] * params['BATCH_SIZE'] / dataset.len_train)
                params['EPOCH_OFFSET'] = params['RELOAD'] if params['RELOAD_EPOCH'] else epoch_offset

                for split, filename in iteritems(params['TEXT_FILES']):
                    dataset = update_dataset_from_file(dataset,
                                                       os.path.join(params['DATA_ROOT_PATH'],
                                                                    filename + params['SRC_LAN']),
                                                       params,
                                                       splits=list([split]),
                                                       output_text_filename=os.path.join(params['DATA_ROOT_PATH'],
                                                                                         filename + params['TRG_LAN']),
                                                       remove_outputs=False,
                                                       compute_state_below=True,
                                                       recompute_references=True)
                    dataset.name = params['DATASET_NAME'] + '_' + params['SRC_LAN'] + params['TRG_LAN']
                saveDataset(dataset, params['DATASET_STORE_PATH'])

        else:
            logger.info('Reloading and using dataset.')
            dataset = loadDataset(load_dataset)
    else:
        # Load data
        if load_dataset is None:
            dataset = build_dataset(params)
        else:
            dataset = loadDataset(load_dataset)

    params['INPUT_VOCABULARY_SIZE'] = dataset.vocabulary_len[params['INPUTS_IDS_DATASET'][0]]
    params['OUTPUT_VOCABULARY_SIZE'] = dataset.vocabulary_len[params['OUTPUTS_IDS_DATASET'][0]]

    # Build model
    set_optimizer = True if params['RELOAD'] == 0 else False
    clear_dirs = True if params['RELOAD'] == 0 else False

    # build new model
    nmt_model = TranslationModel(params,
                                 model_type=params['MODEL_TYPE'],
                                 verbose=params['VERBOSE'],
                                 model_name=params['MODEL_NAME'],
                                 vocabularies=dataset.vocabulary,
                                 store_path=params['STORE_PATH'],
                                 set_optimizer=set_optimizer,
                                 clear_dirs=clear_dirs)

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

    if params['RELOAD'] > 0:
        nmt_model = updateModel(nmt_model, params['STORE_PATH'], params['RELOAD'], reload_epoch=params['RELOAD_EPOCH'])
        nmt_model.setParams(params)
        nmt_model.setOptimizer()
        if params.get('EPOCH_OFFSET') is None:
            params['EPOCH_OFFSET'] = params['RELOAD'] if params['RELOAD_EPOCH'] else \
                int(params['RELOAD'] * params['BATCH_SIZE'] / dataset.len_train)

    # Store configuration as pkl
    dict2pkl(params, os.path.join(params['STORE_PATH'], 'config'))

    # Callbacks
    callbacks = buildCallbacks(params, nmt_model, dataset)

    # Training
    total_start_time = timer()

    logger.debug('Starting training!')
    training_params = {'n_epochs': params['MAX_EPOCH'],
                       'batch_size': params['BATCH_SIZE'],
                       'homogeneous_batches': params['HOMOGENEOUS_BATCHES'],
                       'maxlen': params['MAX_OUTPUT_TEXT_LEN'],
                       'joint_batches': params['JOINT_BATCHES'],
                       'lr_decay': params.get('LR_DECAY', None),  # LR decay parameters
                       'initial_lr': params.get('LR', 1.0),
                       'reduce_each_epochs': params.get('LR_REDUCE_EACH_EPOCHS', True),
                       'start_reduction_on_epoch': params.get('LR_START_REDUCTION_ON_EPOCH', 0),
                       'lr_gamma': params.get('LR_GAMMA', 0.9),
                       'lr_reducer_type': params.get('LR_REDUCER_TYPE', 'linear'),
                       'lr_reducer_exp_base': params.get('LR_REDUCER_EXP_BASE', 0),
                       'lr_half_life': params.get('LR_HALF_LIFE', 50000),
                       'lr_warmup_exp': params.get('WARMUP_EXP', -1.5),
                       'min_lr': params.get('MIN_LR', 1e-9),
                       'epochs_for_save': params['EPOCHS_FOR_SAVE'],
                       'verbose': params['VERBOSE'],
                       'eval_on_sets': None,  # Unsupported for autorreggressive models
                       'n_parallel_loaders': params['PARALLEL_LOADERS'],
                       'extra_callbacks': callbacks,
                       'reload_epoch': params['RELOAD'],
                       'epoch_offset': params.get('EPOCH_OFFSET', 0),
                       'data_augmentation': params['DATA_AUGMENTATION'],
                       'patience': params.get('PATIENCE', 0),  # early stopping parameters
                       'metric_check': params.get('STOP_METRIC', None) if params.get('EARLY_STOP', False) else None,
                       'min_delta': params.get('MIN_DELTA', 0.),
                       'eval_on_epochs': params.get('EVAL_EACH_EPOCHS', True),
                       'each_n_epochs': params.get('EVAL_EACH', 1),
                       'start_eval_on_epoch': params.get('START_EVAL_ON_EPOCH', 0),
                       'n_gpus': params.get('N_GPUS', 1),
                       'tensorboard': params.get('TENSORBOARD', False),
                       'tensorboard_params':
                           {
                               'log_dir': params.get('LOG_DIR', 'tensorboard_logs'),
                               'histogram_freq': params.get('HISTOGRAM_FREQ', 0),
                               'batch_size': params.get('TENSORBOARD_BATCH_SIZE', params['BATCH_SIZE']),
                               'write_graph': params.get('WRITE_GRAPH', True),
                               'write_grads': params.get('WRITE_GRADS', False),
                               'write_images': params.get('WRITE_IMAGES', False),
                               'embeddings_freq': None,
                               'embeddings_layer_names': None,
                               'embeddings_metadata': None,
                               'word_embeddings_labels': None,
                               'update_freq': params.get('UPDATE_FREQ', 'epoch')}
                       }
    nmt_model.trainNet(dataset, training_params)

    total_end_time = timer()
    time_difference = total_end_time - total_start_time
    logger.info('In total is {0:.2f}s = {1:.2f}m'.format(time_difference, time_difference / 60.0))
