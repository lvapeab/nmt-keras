# -*- coding: utf-8 -*-
from __future__ import print_function
from six import iteritems
try:
    import itertools.zip as zip
except ImportError:
    pass

import logging
import os
import sys

from keras.layers import *
from keras.models import model_from_json, Model
from keras.utils import multi_gpu_model
from keras.optimizers import *
from keras.regularizers import l2, AlphaRegularizer
from keras_wrapper.cnn_model import Model_Wrapper
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(__name__)

from regularize import Regularize


def getPositionalEncodingWeights(input_dim, output_dim, name='', verbose=True):
    """
    Obtains fixed sinusoidal embeddings for obtaining the positional encoding.

    :param int input_dim: Input dimension of the embeddings (i.e. vocabulary size).
    :param int output_dim: Embeddings dimension.
    :param str name: Name of the layer
    :param int verbose: Be verbose
    :return: A list with sinusoidal embeddings.
    """

    if verbose > 0:
        logger.info("<<< Obtaining positional encodings of layer " + name + " >>>")
    position_enc = np.array([[pos / np.power(10000, 2. * i / output_dim) for i in range(output_dim)] for pos in range(input_dim)])
    position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
    position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
    return [position_enc]


class TranslationModel(Model_Wrapper):
    """
    Translation model class. Instance of the Model_Wrapper class (see staged_keras_wrapper).

    :param dict params: all hyperparameters of the model.
    :param str model_type: network name type (corresponds to any method defined in the section 'MODELS' of this class).
                 Only valid if 'structure_path' == None.
    :param int verbose: set to 0 if you don't want the model to output informative messages
    :param str structure_path: path to a Keras' model json file.
                          If we speficy this parameter then 'type' will be only an informative parameter.
    :param str weights_path: path to the pre-trained weights file (if None, then it will be initialized according to params)
    :param str model_name: optional name given to the network (if None, then it will be assigned to current time as its name)
    :param dict vocabularies: vocabularies used for word embedding
    :param str store_path: path to the folder where the temporal model packups will be stored
    :param bool set_optimizer: Compile optimizer or not.
    :param bool clear_dirs: Clean model directories or not.
    """

    def __init__(self, params, model_type='Translation_Model', verbose=1, structure_path=None, weights_path=None,
                 model_name=None, vocabularies=None, store_path=None, set_optimizer=True, clear_dirs=True, trainable_est=True, trainable_pred=True):
        """
        Translation_Model object constructor.

        :param params: all hyperparams of the model.
        :param model_type: network name type (corresponds to any method defined in the section 'MODELS' of this class).
                     Only valid if 'structure_path' == None.
        :param verbose: set to 0 if you don't want the model to output informative messages
        :param structure_path: path to a Keras' model json file.
                              If we speficy this parameter then 'type' will be only an informative parameter.
        :param weights_path: path to the pre-trained weights file (if None, then it will be randomly initialized)
        :param model_name: optional name given to the network (if None, then it will be assigned to current time as its name)
        :param vocabularies: vocabularies used for word embedding
        :param store_path: path to the folder where the temporal model packups will be stored
        :param set_optimizer: Compile optimizer or not.
        :param clear_dirs: Clean model directories or not.

        :param trainable_est: Is estimator trainable?
        :param trainable_pred: Is predictor trainable?

        """
        super(TranslationModel, self).__init__(model_type=model_type, model_name=model_name,
                                               silence=verbose == 0, models_path=store_path, inheritance=True)

        self.__toprint = ['_model_type', 'name', 'model_path', 'verbose']

        self.verbose = verbose
        self._model_type = model_type
        self.params = params
        self.vocabularies = vocabularies
        self.ids_inputs = params['INPUTS_IDS_MODEL']
        self.ids_outputs = params['OUTPUTS_IDS_MODEL']
        self.return_alphas = params['COVERAGE_PENALTY'] or params['POS_UNK']
        # Sets the model name and prepares the folders for storing the models
        self.setName(model_name, models_path=store_path, clear_dirs=clear_dirs)
        self.trainable_est = trainable_est
        self.trainable = trainable_pred

        self.use_CuDNN = 'CuDNN' if K.backend() == 'tensorflow' and params.get('USE_CUDNN', True) else ''

        # Prepare source word embedding
        if params['SRC_PRETRAINED_VECTORS'] is not None:
            if self.verbose > 0:
                logger.info("<<< Loading pretrained word vectors from:  " + params['SRC_PRETRAINED_VECTORS'] + " >>>")
            src_word_vectors = np.load(os.path.join(params['SRC_PRETRAINED_VECTORS']), allow_pickle=True).item()
            self.src_embedding_weights = np.random.rand(params['INPUT_VOCABULARY_SIZE'],
                                                        params['SOURCE_TEXT_EMBEDDING_SIZE'])
            for word, index in iteritems(self.vocabularies[self.ids_inputs[0]]['words2idx']):
                if src_word_vectors.get(word) is not None:
                    self.src_embedding_weights[index, :] = src_word_vectors[word]
            self.src_embedding_weights = [self.src_embedding_weights]
            self.src_embedding_weights_trainable = params['SRC_PRETRAINED_VECTORS_TRAINABLE'] and params.get('TRAINABLE_ENCODER', True)
            del src_word_vectors

        else:
            self.src_embedding_weights = None
            self.src_embedding_weights_trainable = params.get('TRAINABLE_ENCODER', True)

        # Prepare target word embedding
        if params['TRG_PRETRAINED_VECTORS'] is not None:
            if self.verbose > 0:
                logger.info("<<< Loading pretrained word vectors from: " + params['TRG_PRETRAINED_VECTORS'] + " >>>")
            trg_word_vectors = np.load(os.path.join(params['TRG_PRETRAINED_VECTORS']), allow_pickle=True).item()
            self.trg_embedding_weights = np.random.rand(params['OUTPUT_VOCABULARY_SIZE'],
                                                        params['TARGET_TEXT_EMBEDDING_SIZE'])
            for word, index in iteritems(self.vocabularies[self.ids_outputs[0]]['words2idx']):
                if trg_word_vectors.get(word) is not None:
                    self.trg_embedding_weights[index, :] = trg_word_vectors[word]
            self.trg_embedding_weights = [self.trg_embedding_weights]
            self.trg_embedding_weights_trainable = params['TRG_PRETRAINED_VECTORS_TRAINABLE'] and params.get('TRAINABLE_DECODER', True)
            del trg_word_vectors
        else:
            self.trg_embedding_weights = None
            self.trg_embedding_weights_trainable = params.get('TRAINABLE_DECODER', True)

        # Prepare model
        if structure_path:
            # Load a .json model
            if self.verbose > 0:
                logger.info("<<< Loading model structure from file " + structure_path + " >>>")
            self.model = model_from_json(open(structure_path).read())
        else:
            # Build model from scratch
            if hasattr(self, model_type):
                if self.verbose > 0:
                    logger.info("<<< Building " + model_type + " Translation_Model >>>")
                eval('self.' + model_type + '(params)')
            else:
                raise Exception('Translation_Model model_type "' + model_type + '" is not implemented.')

        # Load weights from file
        if weights_path:
            if self.verbose > 0:
                logger.info("<<< Loading weights from file " + weights_path + " >>>")
            self.model.load_weights(weights_path, by_name=True)

        # Print information of self
        if verbose > 0:
            print(str(self))
            self.model.summary()
            sys.stdout.flush()

        if set_optimizer:
            self.setOptimizer()

    def setParams(self, params):
        self.params = params

    def setOptimizer(self, **kwargs):
        """
        Sets and compiles a new optimizer for the Translation_Model.
        The configuration is read from Translation_Model.params.
        :return: None
        """
        if int(self.params.get('ACCUMULATE_GRADIENTS', 1)) > 1 and self.params['OPTIMIZER'].lower() != 'adam':
            logger.warning('Gradient accumulate is only implemented for the Adam optimizer. Setting "ACCUMULATE_GRADIENTS" to 1.')
            self.params['ACCUMULATE_GRADIENTS'] = 1

        optimizer_str = '\t LR: ' + str(self.params.get('LR', 0.01)) + \
                        '\n\t LOSS: ' + str(self.params.get('LOSS', 'categorical_crossentropy'))

        if self.params.get('USE_TF_OPTIMIZER', False) and K.backend() == 'tensorflow':
            if self.params['OPTIMIZER'].lower() not in ['sgd', 'adagrad', 'adadelta', 'rmsprop', 'adam']:
                logger.warning('The optimizer %s is not natively implemented in Tensorflow. Using the Keras version.' % (str(self.params['OPTIMIZER'])))
            if self.params.get('LR_DECAY') is not None:
                logger.warning('The learning rate decay is not natively implemented in native Tensorflow optimizers. Using the Keras version.')
                self.params['USE_TF_OPTIMIZER'] = False
            if self.params.get('ACCUMULATE_GRADIENTS', 1) > 1:
                logger.warning('The gradient accumulation is not natively implemented in native Tensorflow optimizers. Using the Keras version.')
                self.params['USE_TF_OPTIMIZER'] = False

        if self.params.get('USE_TF_OPTIMIZER', False) and K.backend() == 'tensorflow' and self.params['OPTIMIZER'].lower() in ['sgd', 'adagrad', 'adadelta', 'rmsprop', 'adam']:
            import tensorflow as tf
            if self.params['OPTIMIZER'].lower() == 'sgd':
                if self.params.get('MOMENTUM') is None:
                    optimizer = TFOptimizer(tf.train.GradientDescentOptimizer(self.params.get('LR', 0.01)))
                else:
                    optimizer = TFOptimizer(tf.train.MomentumOptimizer(self.params.get('LR', 0.01),
                                                                       self.params.get('MOMENTUM', 0.0),
                                                                       use_nesterov=self.params.get('NESTEROV_MOMENTUM', False)))
                    optimizer_str += '\n\t MOMENTUM: ' + str(self.params.get('MOMENTUM', 0.0)) + \
                                     '\n\t NESTEROV: ' + str(self.params.get('NESTEROV_MOMENTUM', False))

            elif self.params['OPTIMIZER'].lower() == 'adam':
                optimizer = TFOptimizer(tf.train.AdamOptimizer(learning_rate=self.params.get('LR', 0.01),
                                                               beta1=self.params.get('BETA_1', 0.9),
                                                               beta2=self.params.get('BETA_2', 0.999),
                                                               epsilon=self.params.get('EPSILON', 1e-7)))
                optimizer_str += '\n\t BETA_1: ' + str(self.params.get('BETA_1', 0.9)) + \
                                 '\n\t BETA_2: ' + str(self.params.get('BETA_2', 0.999)) + \
                                 '\n\t EPSILON: ' + str(self.params.get('EPSILON', 1e-7))

            elif self.params['OPTIMIZER'].lower() == 'adagrad':
                optimizer = TFOptimizer(tf.train.AdagradOptimizer(self.params.get('LR', 0.01)))

            elif self.params['OPTIMIZER'].lower() == 'rmsprop':
                optimizer = TFOptimizer(tf.train.RMSPropOptimizer(self.params.get('LR', 0.01),
                                                                  decay=self.params.get('LR_OPTIMIZER_DECAY', 0.0),
                                                                  momentum=self.params.get('MOMENTUM', 0.0),
                                                                  epsilon=self.params.get('EPSILON', 1e-7)))
                optimizer_str += '\n\t MOMENTUM: ' + str(self.params.get('MOMENTUM', 0.9)) + \
                                 '\n\t EPSILON: ' + str(self.params.get('EPSILON', 1e-7))

            elif self.params['OPTIMIZER'].lower() == 'adadelta':
                optimizer = TFOptimizer(tf.train.AdadeltaOptimizer(learning_rate=self.params.get('LR', 0.01),
                                                                   rho=self.params.get('RHO', 0.95),
                                                                   epsilon=self.params.get('EPSILON', 1e-7)))
                optimizer_str += '\n\t RHO: ' + str(self.params.get('RHO', 0.9)) + \
                                 '\n\t EPSILON: ' + str(self.params.get('EPSILON', 1e-7))

            else:
                raise Exception('\tThe chosen optimizer is not implemented.')
        else:
            if self.params['OPTIMIZER'].lower() == 'sgd':
                optimizer = SGD(lr=self.params.get('LR', 0.01),
                                momentum=self.params.get('MOMENTUM', 0.0),
                                decay=self.params.get('LR_OPTIMIZER_DECAY', 0.0),
                                nesterov=self.params.get('NESTEROV_MOMENTUM', False),
                                clipnorm=self.params.get('CLIP_C', 0.),
                                clipvalue=self.params.get('CLIP_V', 0.))
                optimizer_str += '\n\t MOMENTUM: ' + str(self.params.get('MOMENTUM', 0.0)) + \
                                 '\n\t NESTEROV: ' + str(self.params.get('NESTEROV_MOMENTUM', False))

            elif self.params['OPTIMIZER'].lower() == 'rsmprop':
                optimizer = RMSprop(lr=self.params.get('LR', 0.001),
                                    rho=self.params.get('RHO', 0.9),
                                    decay=self.params.get('LR_OPTIMIZER_DECAY', 0.0),
                                    clipnorm=self.params.get('CLIP_C', 0.),
                                    clipvalue=self.params.get('CLIP_V', 0.),
                                    epsilon=self.params.get('EPSILON', 1e-7))
                optimizer_str += '\n\t RHO: ' + str(self.params.get('RHO', 0.9)) + \
                                 '\n\t EPSILON: ' + str(self.params.get('EPSILON', 1e-7))

            elif self.params['OPTIMIZER'].lower() == 'adagrad':
                optimizer = Adagrad(lr=self.params.get('LR', 0.01),
                                    decay=self.params.get('LR_OPTIMIZER_DECAY', 0.0),
                                    clipnorm=self.params.get('CLIP_C', 0.),
                                    clipvalue=self.params.get('CLIP_V', 0.),
                                    epsilon=self.params.get('EPSILON', 1e-7))
                optimizer_str += '\n\t EPSILON: ' + str(self.params.get('EPSILON', 1e-7))

            elif self.params['OPTIMIZER'].lower() == 'adadelta':
                optimizer = Adadelta(lr=self.params.get('LR', 1.0),
                                     rho=self.params.get('RHO', 0.9),
                                     decay=self.params.get('LR_OPTIMIZER_DECAY', 0.0),
                                     clipnorm=self.params.get('CLIP_C', 0.),
                                     clipvalue=self.params.get('CLIP_V', 0.),
                                     epsilon=self.params.get('EPSILON', 1e-7))
                optimizer_str += '\n\t RHO: ' + str(self.params.get('RHO', 0.9)) + \
                                 '\n\t EPSILON: ' + str(self.params.get('EPSILON', 1e-7))

            elif self.params['OPTIMIZER'].lower() == 'adam':
                if self.params.get('ACCUMULATE_GRADIENTS', 1) > 1:
                    optimizer = AdamAccumulate(lr=self.params.get('LR', 0.001),
                                               beta_1=self.params.get('BETA_1', 0.9),
                                               beta_2=self.params.get('BETA_2', 0.999),
                                               amsgrad=self.params.get('AMSGRAD', False),
                                               decay=self.params.get('LR_OPTIMIZER_DECAY', 0.0),
                                               clipnorm=self.params.get('CLIP_C', 0.),
                                               clipvalue=self.params.get('CLIP_V', 0.),
                                               epsilon=self.params.get('EPSILON', 1e-7),
                                               accum_iters=self.params.get('ACCUMULATE_GRADIENTS'))
                    optimizer_str += '\n\t BETA_1: ' + str(self.params.get('BETA_1', 0.9)) + \
                                     '\n\t BETA_2: ' + str(self.params.get('BETA_2', 0.999)) + \
                                     '\n\t AMSGRAD: ' + str(self.params.get('AMSGRAD', False)) + \
                                     '\n\t ACCUMULATE_GRADIENTS: ' + str(self.params.get('ACCUMULATE_GRADIENTS')) + \
                                     '\n\t EPSILON: ' + str(self.params.get('EPSILON', 1e-7))
                else:
                    optimizer = Adam(lr=self.params.get('LR', 0.001),
                                     beta_1=self.params.get('BETA_1', 0.9),
                                     beta_2=self.params.get('BETA_2', 0.999),
                                     amsgrad=self.params.get('AMSGRAD', False),
                                     decay=self.params.get('LR_OPTIMIZER_DECAY', 0.0),
                                     clipnorm=self.params.get('CLIP_C', 0.),
                                     clipvalue=self.params.get('CLIP_V', 0.),
                                     epsilon=self.params.get('EPSILON', 1e-7))
                    optimizer_str += '\n\t BETA_1: ' + str(self.params.get('BETA_1', 0.9)) + \
                                     '\n\t BETA_2: ' + str(self.params.get('BETA_2', 0.999)) + \
                                     '\n\t AMSGRAD: ' + str(self.params.get('AMSGRAD', False)) + \
                                     '\n\t EPSILON: ' + str(self.params.get('EPSILON', 1e-7))

            elif self.params['OPTIMIZER'].lower() == 'adamax':
                optimizer = Adamax(lr=self.params.get('LR', 0.002),
                                   beta_1=self.params.get('BETA_1', 0.9),
                                   beta_2=self.params.get('BETA_2', 0.999),
                                   decay=self.params.get('LR_OPTIMIZER_DECAY', 0.0),
                                   clipnorm=self.params.get('CLIP_C', 0.),
                                   clipvalue=self.params.get('CLIP_V', 0.),
                                   epsilon=self.params.get('EPSILON', 1e-7))
                optimizer_str += '\n\t BETA_1: ' + str(self.params.get('BETA_1', 0.9)) + \
                                 '\n\t BETA_2: ' + str(self.params.get('BETA_2', 0.999)) + \
                                 '\n\t EPSILON: ' + str(self.params.get('EPSILON', 1e-7))
            elif self.params['OPTIMIZER'].lower() == 'nadam':
                optimizer = Nadam(lr=self.params.get('LR', 0.002),
                                  beta_1=self.params.get('BETA_1', 0.9),
                                  beta_2=self.params.get('BETA_2', 0.999),
                                  schedule_decay=self.params.get('LR_OPTIMIZER_DECAY', 0.0),
                                  clipnorm=self.params.get('CLIP_C', 0.),
                                  clipvalue=self.params.get('CLIP_V', 0.),
                                  epsilon=self.params.get('EPSILON', 1e-7))
                optimizer_str += '\n\t BETA_1: ' + str(self.params.get('BETA_1', 0.9)) + \
                                 '\n\t BETA_2: ' + str(self.params.get('BETA_2', 0.999)) + \
                                 '\n\t EPSILON: ' + str(self.params.get('EPSILON', 1e-7))

            elif self.params['OPTIMIZER'].lower() == 'sgdhd':
                optimizer = SGDHD(lr=self.params.get('LR', 0.002),
                                  clipnorm=self.params.get('CLIP_C', 10.),
                                  clipvalue=self.params.get('CLIP_V', 0.),
                                  hypergrad_lr=self.params.get('HYPERGRAD_LR', 0.001))
                optimizer_str += '\n\t HYPERGRAD_LR: ' + str(self.params.get('HYPERGRAD_LR', 0.001))

            elif self.params['OPTIMIZER'].lower() == 'qhsgd':
                optimizer = QHSGD(lr=self.params.get('LR', 0.002),
                                  momentum=self.params.get('MOMENTUM', 0.0),
                                  quasi_hyperbolic_momentum=self.params.get('QUASI_HYPERBOLIC_MOMENTUM', 0.0),
                                  decay=self.params.get('LR_OPTIMIZER_DECAY', 0.0),
                                  nesterov=self.params.get('NESTEROV_MOMENTUM', False),
                                  dampening=self.params.get('DAMPENING', 0.),
                                  clipnorm=self.params.get('CLIP_C', 10.),
                                  clipvalue=self.params.get('CLIP_V', 0.))
                optimizer_str += '\n\t MOMENTUM: ' + str(self.params.get('MOMENTUM', 0.0)) + \
                                 '\n\t QUASI_HYPERBOLIC_MOMENTUM: ' + str(self.params.get('QUASI_HYPERBOLIC_MOMENTUM', 0.0)) + \
                                 '\n\t DAMPENING: ' + str(self.params.get('DAMPENING', 0.0)) + \
                                 '\n\t NESTEROV: ' + str(self.params.get('NESTEROV_MOMENTUM', False))

            elif self.params['OPTIMIZER'].lower() == 'qhsgdhd':
                optimizer = QHSGDHD(lr=self.params.get('LR', 0.002),
                                    momentum=self.params.get('MOMENTUM', 0.0),
                                    quasi_hyperbolic_momentum=self.params.get('QUASI_HYPERBOLIC_MOMENTUM', 0.0),
                                    dampening=self.params.get('DAMPENING', 0.),
                                    hypergrad_lr=self.params.get('HYPERGRAD_LR', 0.001),
                                    decay=self.params.get('LR_OPTIMIZER_DECAY', 0.0),
                                    nesterov=self.params.get('NESTEROV_MOMENTUM', False),
                                    clipnorm=self.params.get('CLIP_C', 10.),
                                    clipvalue=self.params.get('CLIP_V', 0.))
                optimizer_str += '\n\t MOMENTUM: ' + str(self.params.get('MOMENTUM', 0.0)) + \
                                 '\n\t QUASI_HYPERBOLIC_MOMENTUM: ' + str(self.params.get('QUASI_HYPERBOLIC_MOMENTUM', 0.0)) + \
                                 '\n\t HYPERGRAD_LR: ' + str(self.params.get('HYPERGRAD_LR', 0.001)) + \
                                 '\n\t DAMPENING: ' + str(self.params.get('DAMPENING', 0.0)) + \
                                 '\n\t NESTEROV: ' + str(self.params.get('NESTEROV_MOMENTUM', False))

            elif self.params['OPTIMIZER'].lower() == 'adadeltahd':
                optimizer = AdadeltaHD(lr=self.params.get('LR', 0.002),
                                       hypergrad_lr=self.params.get('HYPERGRAD_LR', 0.001),
                                       rho=self.params.get('RHO', 0.9),
                                       decay=self.params.get('LR_OPTIMIZER_DECAY', 0.0),
                                       epsilon=self.params.get('EPSILON', 1e-7),
                                       clipnorm=self.params.get('CLIP_C', 10.),
                                       clipvalue=self.params.get('CLIP_V', 0.))
                optimizer_str += '\n\t HYPERGRAD_LR: ' + str(self.params.get('HYPERGRAD_LR', 0.001)) + \
                                 '\n\t RHO: ' + str(self.params.get('RHO', 0.9)) + \
                                 '\n\t EPSILON: ' + str(self.params.get('EPSILON', 1e-7))

            elif self.params['OPTIMIZER'].lower() == 'adamhd':
                optimizer = AdamHD(lr=self.params.get('LR', 0.002),
                                   hypergrad_lr=self.params.get('HYPERGRAD_LR', 0.001),
                                   beta_1=self.params.get('BETA_1', 0.9),
                                   beta_2=self.params.get('BETA_2', 0.999),
                                   decay=self.params.get('LR_OPTIMIZER_DECAY', 0.0),
                                   clipnorm=self.params.get('CLIP_C', 10.),
                                   clipvalue=self.params.get('CLIP_V', 0.),
                                   epsilon=self.params.get('EPSILON', 1e-7))
                optimizer_str += '\n\t HYPERGRAD_LR: ' + str(self.params.get('HYPERGRAD_LR', 0.001)) + \
                                 '\n\t BETA_1: ' + str(self.params.get('BETA_1', 0.9)) + \
                                 '\n\t BETA_2: ' + str(self.params.get('BETA_2', 0.999)) + \
                                 '\n\t EPSILON: ' + str(self.params.get('EPSILON', 1e-7))
            else:
                logger.info('\tWARNING: The modification of the LR is not implemented for the chosen optimizer.')
                optimizer = eval(self.params['OPTIMIZER'])

            optimizer_str += '\n\t CLIP_C ' + str(self.params.get('CLIP_C', 0.)) + \
                             '\n\t CLIP_V ' + str(self.params.get('CLIP_V', 0.)) + \
                             '\n\t LR_OPTIMIZER_DECAY ' + str(self.params.get('LR_OPTIMIZER_DECAY', 0.0)) + \
                             '\n\t ACCUMULATE_GRADIENTS ' + str(self.params.get('ACCUMULATE_GRADIENTS', 1)) + '\n'
        if self.verbose > 0:
            logger.info("Preparing optimizer and compiling. Optimizer configuration: \n" + optimizer_str)


        sample_weight_mode = []
        sample_weight_dict = self.params['SAMPLE_WEIGHTS'] 

        for out_id in self.ids_outputs:

            if out_id in sample_weight_dict:
                sample_weight_mode.append('temporal')
            else:
                sample_weight_mode.append(None)  
        

        if hasattr(self, 'multi_gpu_model') and self.multi_gpu_model is not None:
            model_to_compile = self.multi_gpu_model
        else:
            model_to_compile = self.model

        model_to_compile.compile(optimizer=optimizer,
                                 loss=self.params['LOSS'],
                                 metrics=self.params.get('KERAS_METRICS', []),
                                 loss_weights=self.params.get('LOSS_WEIGHTS', None),
                                 sample_weight_mode=sample_weight_mode,
                                 weighted_metrics=self.params.get('KERAS_METRICS_WEIGHTS', None),
                                 target_tensors=self.params.get('TARGET_TENSORS'))

    def __str__(self):
        """
        Plots basic model information.

        :return: String containing model information.
        """
        obj_str = '-----------------------------------------------------------------------------------\n'
        class_name = self.__class__.__name__
        obj_str += '\t\t' + class_name + ' instance\n'
        obj_str += '-----------------------------------------------------------------------------------\n'

        # Print pickled attributes
        for att in self.__toprint:
            obj_str += att + ': ' + str(self.__dict__[att])
            obj_str += '\n'

        obj_str += '\n'
        obj_str += 'Params:\n\t'
        obj_str += "\n\t".join([str(key) + ": " + str(self.params[key]) for key in sorted(self.params.keys())])
        obj_str += '\n'
        obj_str += '-----------------------------------------------------------------------------------'

        return obj_str

    # ------------------------------------------------------- #
    #       PREDEFINED MODELS
    # ------------------------------------------------------- #

    def AttentionRNNEncoderDecoder(self, params):
        """
        Neural machine translation with:
            * BRNN encoder
            * Attention mechansim on input sequence of annotations
            * Conditional RNN for decoding
            * Deep output layers:
            * Context projected to output
            * Last word projected to output
            * Possibly deep encoder/decoder
        See:
            * `Neural Machine Translation by Jointly Learning to Align and Translate`_.
            * `Nematus\: a Toolkit for Neural Machine Translation`_.

        .. _Neural Machine Translation by Jointly Learning to Align and Translate: https://arxiv.org/abs/1409.0473
        .. _Nematus\: a Toolkit for Neural Machine Translation: https://arxiv.org/abs/1703.04357

        :param int params: Dictionary of hyper-params (see config.py)
        :return: None
        """

        # 1. Source text input
        src_text = Input(name=self.ids_inputs[0], batch_shape=tuple([None, None]), dtype='int32')
        # 2. Encoder
        # 2.1. Source word embedding
        embedding = Embedding(params['INPUT_VOCABULARY_SIZE'], params['SOURCE_TEXT_EMBEDDING_SIZE'],
                              name='source_word_embedding',
                              embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                              embeddings_initializer=params['INIT_FUNCTION'],
                              trainable=self.src_embedding_weights_trainable,
                              weights=self.src_embedding_weights,
                              mask_zero=True)
        src_embedding = embedding(src_text)

        if params.get('SCALE_SOURCE_WORD_EMBEDDINGS', False):
            src_embedding = SqrtScaling(params['SOURCE_TEXT_EMBEDDING_SIZE'])(src_embedding)

        src_embedding = Regularize(src_embedding, params, name='src_embedding')

        # Get mask of source embeddings (CuDNN RNNs don't accept masks)
        src_embedding_mask = GetMask(name='source_text_mask')(src_embedding)
        src_embedding = RemoveMask()(src_embedding)

        if params['RECURRENT_INPUT_DROPOUT_P'] > 0.:
            src_embedding = Dropout(params['RECURRENT_INPUT_DROPOUT_P'])(src_embedding)

        # 2.2. BRNN encoder (GRU/LSTM)
        if params['BIDIRECTIONAL_ENCODER']:
            annotations = Bidirectional(eval(self.use_CuDNN + params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                                          kernel_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                                                          recurrent_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                                                          bias_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                                                          kernel_initializer=params['INIT_FUNCTION'],
                                                                                          recurrent_initializer=params['INNER_INIT'],
                                                                                          trainable=params.get('TRAINABLE_ENCODER', True),
                                                                                          return_sequences=True),
                                        trainable=params.get('TRAINABLE_ENCODER', True),
                                        name='bidirectional_encoder_' + params['ENCODER_RNN_TYPE'],
                                        merge_mode=params.get('BIDIRECTIONAL_MERGE_MODE', 'concat'))(src_embedding)
        else:
            annotations = eval(self.use_CuDNN + params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                            kernel_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                                            recurrent_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                                            bias_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                                            kernel_initializer=params['INIT_FUNCTION'],
                                                                            recurrent_initializer=params['INNER_INIT'],
                                                                            trainable=params.get('TRAINABLE_ENCODER', True),
                                                                            return_sequences=True,
                                                                            name='encoder_' + params['ENCODER_RNN_TYPE'])(src_embedding)
        annotations = Regularize(annotations, params, name='annotations')
        # 2.3. Potentially deep encoder
        for n_layer in range(1, params['N_LAYERS_ENCODER']):

            if params['RECURRENT_INPUT_DROPOUT_P'] > 0.:
                annotations = Dropout(params['RECURRENT_INPUT_DROPOUT_P'])(annotations)

            if params['BIDIRECTIONAL_DEEP_ENCODER']:
                current_annotations = Bidirectional(eval(self.use_CuDNN + params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                                                      kernel_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                                                                      recurrent_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                                                                      bias_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                                                                      kernel_initializer=params['INIT_FUNCTION'],
                                                                                                      recurrent_initializer=params['INNER_INIT'],
                                                                                                      trainable=params.get('TRAINABLE_ENCODER', True),
                                                                                                      return_sequences=True),
                                                    merge_mode=params.get('BIDIRECTIONAL_MERGE_MODE', 'concat'),
                                                    trainable=params.get('TRAINABLE_ENCODER', True),
                                                    name='bidirectional_encoder_' + str(n_layer))(annotations)
                current_annotations = Regularize(current_annotations, params, name='annotations_' + str(n_layer))
                annotations = current_annotations if n_layer == 1 and not params['BIDIRECTIONAL_ENCODER'] else Add()([annotations, current_annotations])
            else:
                current_annotations = eval(self.use_CuDNN + params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                                        kernel_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                                                        recurrent_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                                                        bias_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                                                        kernel_initializer=params['INIT_FUNCTION'],
                                                                                        recurrent_initializer=params['INNER_INIT'],
                                                                                        return_sequences=True,
                                                                                        trainable=params.get('TRAINABLE_ENCODER', True),
                                                                                        name='encoder_' + str(n_layer))(annotations)

                current_annotations = Regularize(current_annotations, params, name='annotations_' + str(n_layer))
                annotations = current_annotations if n_layer == 1 and params['BIDIRECTIONAL_ENCODER'] else Add()([annotations, current_annotations])

        # 3. Decoder
        # 3.1.1. Previously generated words as inputs for training -> Teacher forcing
        next_words = Input(name=self.ids_inputs[1], batch_shape=tuple([None, None]), dtype='int32')
        # 3.1.2. Target word embedding
        if params.get('TIE_EMBEDDINGS', False):
            state_below = embedding(next_words)
        else:
            state_below = Embedding(params['OUTPUT_VOCABULARY_SIZE'], params['TARGET_TEXT_EMBEDDING_SIZE'],
                                    name='target_word_embedding',
                                    embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                    embeddings_initializer=params['INIT_FUNCTION'],
                                    trainable=self.trg_embedding_weights_trainable,
                                    weights=self.trg_embedding_weights,
                                    mask_zero=True)(next_words)

        if params.get('SCALE_TARGET_WORD_EMBEDDINGS', False):
            state_below = SqrtScaling(params['TARGET_TEXT_EMBEDDING_SIZE'])(state_below)
        state_below = Regularize(state_below, params, name='state_below')

        # 3.2. Decoder's RNN initialization perceptrons with ctx mean
        annotations = ApplyMask(name='annotations')([annotations, src_embedding_mask])  # We may want the padded annotations
        ctx_mean = MaskedMean(name='ctx_mean')(annotations)

        if len(params['INIT_LAYERS']) > 0:
            for n_layer_init in range(len(params['INIT_LAYERS']) - 1):
                ctx_mean = Dense(params['DECODER_HIDDEN_SIZE'], name='init_layer_%d' % n_layer_init,
                                 kernel_initializer=params['INIT_FUNCTION'],
                                 kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                 bias_regularizer=l2(params['WEIGHT_DECAY']),
                                 trainable=params.get('TRAINABLE_DECODER', True),
                                 activation=params['INIT_LAYERS'][n_layer_init]
                                 )(ctx_mean)
                ctx_mean = Regularize(ctx_mean, params, name='ctx' + str(n_layer_init))

            initial_state = Dense(params['DECODER_HIDDEN_SIZE'], name='initial_state',
                                  kernel_initializer=params['INIT_FUNCTION'],
                                  kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                  bias_regularizer=l2(params['WEIGHT_DECAY']),
                                  trainable=params.get('TRAINABLE_DECODER', True),
                                  activation=params['INIT_LAYERS'][-1]
                                  )(ctx_mean)
            initial_state = Regularize(initial_state, params, name='initial_state')
            input_attentional_decoder = [state_below, annotations, initial_state]

            if 'LSTM' in params['DECODER_RNN_TYPE']:
                initial_memory = Dense(params['DECODER_HIDDEN_SIZE'], name='initial_memory',
                                       kernel_initializer=params['INIT_FUNCTION'],
                                       kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                       bias_regularizer=l2(params['WEIGHT_DECAY']),
                                       trainable=params.get('TRAINABLE_DECODER', True),
                                       activation=params['INIT_LAYERS'][-1])(ctx_mean)
                initial_memory = Regularize(initial_memory, params, name='initial_memory')
                input_attentional_decoder.append(initial_memory)
        else:
            # Initialize to zeros vector
            input_attentional_decoder = [state_below, annotations]
            initial_state = ZeroesLayer(params['DECODER_HIDDEN_SIZE'])(ctx_mean)
            input_attentional_decoder.append(initial_state)
            if 'LSTM' in params['DECODER_RNN_TYPE']:
                input_attentional_decoder.append(initial_state)

        # 3.3. Attentional decoder
        sharedAttRNNCond = eval('Att' + params['DECODER_RNN_TYPE'] + 'Cond')(params['DECODER_HIDDEN_SIZE'],
                                                                             attention_mode=params.get('ATTENTION_MODE', 'add'),
                                                                             att_units=params.get('ATTENTION_SIZE', 0),
                                                                             kernel_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                                             recurrent_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                                             conditional_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                                             bias_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                                             attention_context_wa_regularizer=l2(params['WEIGHT_DECAY']),
                                                                             attention_recurrent_regularizer=l2(params['WEIGHT_DECAY']),
                                                                             attention_context_regularizer=l2(params['WEIGHT_DECAY']),
                                                                             bias_ba_regularizer=l2(params['WEIGHT_DECAY']),
                                                                             dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                                             recurrent_dropout=params['RECURRENT_DROPOUT_P'],
                                                                             conditional_dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                                             attention_dropout=params.get('ATTENTION_DROPOUT_P', 0.),
                                                                             kernel_initializer=params['INIT_FUNCTION'],
                                                                             recurrent_initializer=params['INNER_INIT'],
                                                                             attention_context_initializer=params['INIT_ATT'],
                                                                             trainable=params.get('TRAINABLE_DECODER', True),
                                                                             return_sequences=True,
                                                                             return_extra_variables=True,
                                                                             return_states=True,
                                                                             num_inputs=len(input_attentional_decoder),
                                                                             name='decoder_Att' + params['DECODER_RNN_TYPE'] + 'Cond')

        rnn_output = sharedAttRNNCond(input_attentional_decoder)
        proj_h = rnn_output[0]
        x_att = rnn_output[1]
        alphas = rnn_output[2]
        h_state = rnn_output[3]
        if 'LSTM' in params['DECODER_RNN_TYPE']:
            h_memory = rnn_output[4]
        shared_Lambda_Permute = PermuteGeneral((1, 0, 2))

        if params['DOUBLE_STOCHASTIC_ATTENTION_REG'] > 0:
            alpha_regularizer = AlphaRegularizer(alpha_factor=params['DOUBLE_STOCHASTIC_ATTENTION_REG'])(alphas)

        [proj_h, shared_reg_proj_h] = Regularize(proj_h, params, shared_layers=True, name='proj_h0')

        # 3.4. Possibly deep decoder
        shared_proj_h_list = []
        shared_reg_proj_h_list = []

        h_states_list = [h_state]
        if 'LSTM' in params['DECODER_RNN_TYPE']:
            h_memories_list = [h_memory]

        for n_layer in range(1, params['N_LAYERS_DECODER']):
            current_rnn_input = [proj_h, shared_Lambda_Permute(x_att), initial_state]
            if 'LSTM' in params['DECODER_RNN_TYPE']:
                current_rnn_input.append(initial_memory)
            shared_proj_h_list.append(eval(params['DECODER_RNN_TYPE'].replace('Conditional', '') + 'Cond')(
                params['DECODER_HIDDEN_SIZE'],
                kernel_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                recurrent_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                conditional_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                bias_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                dropout=params['RECURRENT_DROPOUT_P'],
                recurrent_dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                conditional_dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                kernel_initializer=params['INIT_FUNCTION'],
                recurrent_initializer=params['INNER_INIT'],
                return_sequences=True,
                return_states=True,
                trainable=params.get('TRAINABLE_DECODER', True),
                num_inputs=len(current_rnn_input),
                name='decoder_' + params['DECODER_RNN_TYPE'].replace('Conditional', '') + 'Cond' + str(n_layer)))

            current_rnn_output = shared_proj_h_list[-1](current_rnn_input)
            current_proj_h = current_rnn_output[0]
            h_states_list.append(current_rnn_output[1])
            if 'LSTM' in params['DECODER_RNN_TYPE']:
                h_memories_list.append(current_rnn_output[2])
            [current_proj_h, shared_reg_proj_h] = Regularize(current_proj_h, params, shared_layers=True,
                                                             name='proj_h' + str(n_layer))
            shared_reg_proj_h_list.append(shared_reg_proj_h)

            proj_h = Add()([proj_h, current_proj_h])

        # 3.5. Skip connections between encoder and output layer
        shared_FC_mlp = TimeDistributed(Dense(params['SKIP_VECTORS_HIDDEN_SIZE'],
                                              kernel_initializer=params['INIT_FUNCTION'],
                                              kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                              bias_regularizer=l2(params['WEIGHT_DECAY']),
                                              trainable=params.get('TRAINABLE_DECODER', True),
                                              activation='linear'),
                                        trainable=params.get('TRAINABLE_DECODER', True),
                                        name='logit_lstm')
        out_layer_mlp = shared_FC_mlp(proj_h)
        shared_FC_ctx = TimeDistributed(Dense(params['SKIP_VECTORS_HIDDEN_SIZE'],
                                              kernel_initializer=params['INIT_FUNCTION'],
                                              kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                              bias_regularizer=l2(params['WEIGHT_DECAY']),
                                              trainable=params.get('TRAINABLE_DECODER', True),
                                              activation='linear'),
                                        trainable=params.get('TRAINABLE_DECODER', True),
                                        name='logit_ctx')
        out_layer_ctx = shared_FC_ctx(x_att)
        out_layer_ctx = shared_Lambda_Permute(out_layer_ctx)
        shared_FC_emb = TimeDistributed(Dense(params['SKIP_VECTORS_HIDDEN_SIZE'],
                                              kernel_initializer=params['INIT_FUNCTION'],
                                              kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                              bias_regularizer=l2(params['WEIGHT_DECAY']),
                                              trainable=params.get('TRAINABLE_DECODER', True),
                                              activation='linear'),
                                        trainable=params.get('TRAINABLE_DECODER', True),
                                        name='logit_emb')
        out_layer_emb = shared_FC_emb(state_below)

        [out_layer_mlp, shared_reg_out_layer_mlp] = Regularize(out_layer_mlp, params,
                                                               shared_layers=True, name='out_layer_mlp')
        [out_layer_ctx, shared_reg_out_layer_ctx] = Regularize(out_layer_ctx, params,
                                                               shared_layers=True, name='out_layer_ctx')
        [out_layer_emb, shared_reg_out_layer_emb] = Regularize(out_layer_emb, params,
                                                               shared_layers=True, name='out_layer_emb')

        shared_additional_output_merge = eval(params['ADDITIONAL_OUTPUT_MERGE_MODE'])(name='additional_input')
        additional_output = shared_additional_output_merge([out_layer_mlp, out_layer_ctx, out_layer_emb])
        shared_activation = Activation(params.get('SKIP_VECTORS_SHARED_ACTIVATION', 'tanh'))

        out_layer = shared_activation(additional_output)

        shared_deep_list = []
        shared_reg_deep_list = []
        # 3.6 Optional deep ouput layer
        for i, (activation, dimension) in enumerate(params['DEEP_OUTPUT_LAYERS']):
            shared_deep_list.append(TimeDistributed(Dense(dimension, activation=activation,
                                                          kernel_initializer=params['INIT_FUNCTION'],
                                                          kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                                          bias_regularizer=l2(params['WEIGHT_DECAY']),
                                                          trainable=params.get('TRAINABLE_DECODER', True),
                                                          ),
                                                    trainable=params.get('TRAINABLE_DECODER', True),
                                                    name=activation + '_%d' % i))
            out_layer = shared_deep_list[-1](out_layer)
            [out_layer, shared_reg_out_layer] = Regularize(out_layer,
                                                           params, shared_layers=True,
                                                           name='out_layer_' + str(activation) + '_%d' % i)
            shared_reg_deep_list.append(shared_reg_out_layer)

        # 3.7. Output layer: Softmax
        shared_FC_soft = TimeDistributed(Dense(params['OUTPUT_VOCABULARY_SIZE'],
                                               activation=params['CLASSIFIER_ACTIVATION'],
                                               kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                               bias_regularizer=l2(params['WEIGHT_DECAY']),
                                               trainable=(params.get('TRAINABLE_DECODER', True) or params.get('TRAIN_ONLY_LAST_LAYER', True)),
                                               ),
                                         trainable=(params.get('TRAINABLE_DECODER', True) or params.get('TRAIN_ONLY_LAST_LAYER', True)),
                                         name=self.ids_outputs[0])
        softout = shared_FC_soft(out_layer)

        self.model = Model(inputs=[src_text, next_words],
                           outputs=softout,
                           name=self.name + '_training')

        if params['DOUBLE_STOCHASTIC_ATTENTION_REG'] > 0.:
            self.model.add_loss(alpha_regularizer)

        if params.get('N_GPUS', 1) > 1:
            self.multi_gpu_model = multi_gpu_model(self.model, gpus=params['N_GPUS'])
        else:
            self.multi_gpu_model = None

        ##################################################################
        #                         SAMPLING MODEL                         #
        ##################################################################
        # Now that we have the basic training model ready, let's prepare the model for applying decoding
        # The beam-search model will include all the minimum required set of layers (decoder stage) which offer the
        # possibility to generate the next state in the sequence given a pre-processed input (encoder stage)
        # First, we need a model that outputs the preprocessed input + initial h state
        # for applying the initial forward pass
        model_init_input = [src_text, next_words]
        model_init_output = [softout, annotations] + h_states_list
        if 'LSTM' in params['DECODER_RNN_TYPE']:
            model_init_output += h_memories_list
        if self.return_alphas:
            model_init_output.append(alphas)
        self.model_init = Model(inputs=model_init_input,
                                outputs=model_init_output,
                                name=self.name + '_model_init')

        # Store inputs and outputs names for model_init
        self.ids_inputs_init = self.ids_inputs
        ids_states_names = ['next_state_' + str(i) for i in range(len(h_states_list))]

        # first output must be the output probs.
        self.ids_outputs_init = self.ids_outputs + ['preprocessed_input'] + ids_states_names
        if 'LSTM' in params['DECODER_RNN_TYPE']:
            ids_memories_names = ['next_memory_' + str(i) for i in range(len(h_memories_list))]
            self.ids_outputs_init += ids_memories_names
        # Second, we need to build an additional model with the capability to have the following inputs:
        #   - preprocessed_input
        #   - prev_word
        #   - prev_state
        # and the following outputs:
        #   - softmax probabilities
        #   - next_state
        preprocessed_size = params['ENCODER_HIDDEN_SIZE'] * 2 if \
            (params['BIDIRECTIONAL_ENCODER'] and params['N_LAYERS_ENCODER'] == 1) or (params['BIDIRECTIONAL_DEEP_ENCODER'] and params['N_LAYERS_ENCODER'] > 1) \
            else params['ENCODER_HIDDEN_SIZE']
        # Define inputs
        n_deep_decoder_layer_idx = 0
        preprocessed_annotations = Input(name='preprocessed_input', shape=tuple([None, preprocessed_size]))
        prev_h_states_list = [Input(name='prev_state_' + str(i),
                                    shape=tuple([params['DECODER_HIDDEN_SIZE']]))
                              for i in range(len(h_states_list))]

        input_attentional_decoder = [state_below, preprocessed_annotations,
                                     prev_h_states_list[n_deep_decoder_layer_idx]]

        if 'LSTM' in params['DECODER_RNN_TYPE']:
            prev_h_memories_list = [Input(name='prev_memory_' + str(i),
                                          shape=tuple([params['DECODER_HIDDEN_SIZE']]))
                                    for i in range(len(h_memories_list))]

            input_attentional_decoder.append(prev_h_memories_list[n_deep_decoder_layer_idx])
        # Apply decoder
        rnn_output = sharedAttRNNCond(input_attentional_decoder)
        proj_h = rnn_output[0]
        x_att = rnn_output[1]
        alphas = rnn_output[2]
        h_states_list = [rnn_output[3]]
        if 'LSTM' in params['DECODER_RNN_TYPE']:
            h_memories_list = [rnn_output[4]]
        for reg in shared_reg_proj_h:
            proj_h = reg(proj_h)

        for (rnn_decoder_layer, proj_h_reg) in zip(shared_proj_h_list, shared_reg_proj_h_list):
            n_deep_decoder_layer_idx += 1
            input_rnn_decoder_layer = [proj_h, shared_Lambda_Permute(x_att),
                                       prev_h_states_list[n_deep_decoder_layer_idx]]
            if 'LSTM' in params['DECODER_RNN_TYPE']:
                input_rnn_decoder_layer.append(prev_h_memories_list[n_deep_decoder_layer_idx])

            current_rnn_output = rnn_decoder_layer(input_rnn_decoder_layer)
            current_proj_h = current_rnn_output[0]
            h_states_list.append(current_rnn_output[1])  # h_state
            if 'LSTM' in params['DECODER_RNN_TYPE']:
                h_memories_list.append(current_rnn_output[2])  # h_memory
            for reg in proj_h_reg:
                current_proj_h = reg(current_proj_h)
            proj_h = Add()([proj_h, current_proj_h])
        out_layer_mlp = shared_FC_mlp(proj_h)
        out_layer_ctx = shared_FC_ctx(x_att)
        out_layer_ctx = shared_Lambda_Permute(out_layer_ctx)
        out_layer_emb = shared_FC_emb(state_below)

        for (reg_out_layer_mlp, reg_out_layer_ctx, reg_out_layer_emb) in zip(shared_reg_out_layer_mlp,
                                                                             shared_reg_out_layer_ctx,
                                                                             shared_reg_out_layer_emb):
            out_layer_mlp = reg_out_layer_mlp(out_layer_mlp)
            out_layer_ctx = reg_out_layer_ctx(out_layer_ctx)
            out_layer_emb = reg_out_layer_emb(out_layer_emb)

        additional_output = shared_additional_output_merge([out_layer_mlp, out_layer_ctx, out_layer_emb])
        out_layer = shared_activation(additional_output)

        for (deep_out_layer, reg_list) in zip(shared_deep_list, shared_reg_deep_list):
            out_layer = deep_out_layer(out_layer)
            for reg in reg_list:
                out_layer = reg(out_layer)

        # Softmax
        softout = shared_FC_soft(out_layer)
        model_next_inputs = [next_words, preprocessed_annotations] + prev_h_states_list
        model_next_outputs = [softout, preprocessed_annotations] + h_states_list
        if 'LSTM' in params['DECODER_RNN_TYPE']:
            model_next_inputs += prev_h_memories_list
            model_next_outputs += h_memories_list

        if self.return_alphas:
            model_next_outputs.append(alphas)

        self.model_next = Model(inputs=model_next_inputs,
                                outputs=model_next_outputs,
                                name=self.name + '_model_next')
        # Store inputs and outputs names for model_next
        # first input must be previous word
        self.ids_inputs_next = [self.ids_inputs[1]] + ['preprocessed_input']
        # first output must be the output probs.
        self.ids_outputs_next = self.ids_outputs + ['preprocessed_input']
        # Input -> Output matchings from model_init to model_next and from model_next to model_next
        self.matchings_init_to_next = {'preprocessed_input': 'preprocessed_input'}
        self.matchings_next_to_next = {'preprocessed_input': 'preprocessed_input'}
        # append all next states and matchings

        for n_state in range(len(prev_h_states_list)):
            self.ids_inputs_next.append('prev_state_' + str(n_state))
            self.ids_outputs_next.append('next_state_' + str(n_state))
            self.matchings_init_to_next['next_state_' + str(n_state)] = 'prev_state_' + str(n_state)
            self.matchings_next_to_next['next_state_' + str(n_state)] = 'prev_state_' + str(n_state)

        if 'LSTM' in params['DECODER_RNN_TYPE']:
            for n_memory in range(len(prev_h_memories_list)):
                self.ids_inputs_next.append('prev_memory_' + str(n_memory))
                self.ids_outputs_next.append('next_memory_' + str(n_memory))
                self.matchings_init_to_next['next_memory_' + str(n_memory)] = 'prev_memory_' + str(n_memory)
                self.matchings_next_to_next['next_memory_' + str(n_memory)] = 'prev_memory_' + str(n_memory)

    def Transformer(self, params):
        """
        Neural machine translation consisting in stacking blocks of:
            * Multi-head attention.
            * Dropout.
            * Residual connection.
            * Normalization.
            * Position-wise feed-forward networks.

        Positional information is injected to the model via embeddings with positional encoding.

        See:
            * `Attention Is All You Need`_.

        .. _Attention Is All You Need: https://arxiv.org/abs/1706.03762

        :param int params: Dictionary of params (see config.py)
        :return: None
        """

        # 1. Source text input
        src_text = Input(name=self.ids_inputs[0], batch_shape=tuple([None, None]), dtype='int32')
        src_positions = PositionLayer(name='position_layer_src_text')(src_text)

        # 2. Encoder
        # 2.1. Source word embedding
        embedding = Embedding(params['INPUT_VOCABULARY_SIZE'], params['SOURCE_TEXT_EMBEDDING_SIZE'],
                              name='source_word_embedding',
                              embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                              embeddings_initializer=params['INIT_FUNCTION'],
                              trainable=self.src_embedding_weights_trainable,
                              weights=self.src_embedding_weights,
                              mask_zero=True)
        src_embedding = embedding(src_text)

        if params.get('SCALE_SOURCE_WORD_EMBEDDINGS', False):
            src_embedding = SqrtScaling(params['MODEL_SIZE'])(src_embedding)
        if params['TARGET_TEXT_EMBEDDING_SIZE'] == params['SOURCE_TEXT_EMBEDDING_SIZE']:
            max_len = max(params['MAX_INPUT_TEXT_LEN'], params['MAX_OUTPUT_TEXT_LEN'], params['MAX_OUTPUT_TEXT_LEN_TEST'])
        else:
            max_len = params['MAX_INPUT_TEXT_LEN']

        positional_embedding = Embedding(max_len,
                                         params['SOURCE_TEXT_EMBEDDING_SIZE'],
                                         name='positional_src_word_embedding',
                                         trainable=False,
                                         weights=getPositionalEncodingWeights(max_len,
                                                                              params['SOURCE_TEXT_EMBEDDING_SIZE'],
                                                                              name='positional_src_word_embedding',
                                                                              verbose=self.verbose))
        positional_src_embedding = positional_embedding(src_positions)
        src_residual_multihead = Add(name='add_src_embedding_positional_src_embedding')([src_embedding, positional_src_embedding])

        # Regularize
        src_residual_multihead = Dropout(params['DROPOUT_P'])(src_residual_multihead)

        prev_src_residual_multihead = src_residual_multihead

        # Left tranformer block (encoder)
        for n_block in range(params['N_LAYERS_ENCODER']):
            src_multihead = MultiHeadAttention(params['N_HEADS'],
                                               params['MODEL_SIZE'],
                                               activation=params.get('MULTIHEAD_ATTENTION_ACTIVATION', 'relu'),
                                               use_bias=True,
                                               dropout=params.get('ATTENTION_DROPOUT_P', 0.),
                                               name='src_MultiHeadAttention_' + str(n_block))([src_residual_multihead,
                                                                                               src_residual_multihead])
            # Regularize
            src_multihead = Dropout(params['DROPOUT_P'])(src_multihead)
            # Add
            src_multihead = Add(name='src_Residual_MultiHeadAttention_' + str(n_block))([src_multihead, prev_src_residual_multihead])

            # And norm
            src_multihead = BatchNormalization(mode=1, name='src_Normalization_MultiHeadAttention_' + str(n_block))(src_multihead)

            # FF
            ff_src_multihead = TimeDistributed(PositionwiseFeedForwardDense(params['FF_SIZE']))(src_multihead)
            # Regularize
            ff_src_multihead = Dropout(params['DROPOUT_P'])(ff_src_multihead)

            # Add
            src_multihead = Add(name='src_Residual_FF_' + str(n_block))([ff_src_multihead, src_multihead])
            # And norm
            src_multihead = BatchNormalization(mode=1, name='src_Normalization_FF_' + str(n_block))(src_multihead)

            prev_src_residual_multihead = src_multihead
            src_residual_multihead = src_multihead

        masked_src_multihead = MaskLayer()(prev_src_residual_multihead)  # We may want the padded annotations

        # 3.1.1. Previously generated words as inputs for training -> Teacher forcing
        next_words = Input(name=self.ids_inputs[1], batch_shape=tuple([None, None]), dtype='int32')
        next_words_positions = PositionLayer(name='position_layer_next_words')(next_words)

        # 3.1.2. Target word embedding
        if params.get('TIE_EMBEDDINGS', False):
            state_below = embedding(next_words)
        else:
            state_below = Embedding(params['OUTPUT_VOCABULARY_SIZE'], params['TARGET_TEXT_EMBEDDING_SIZE'],
                                    name='target_word_embedding',
                                    embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                    embeddings_initializer=params['INIT_FUNCTION'],
                                    trainable=self.trg_embedding_weights_trainable,
                                    weights=self.trg_embedding_weights,
                                    mask_zero=True)(next_words)

        if params.get('SCALE_TARGET_WORD_EMBEDDINGS', False):
            state_below = SqrtScaling(params['MODEL_SIZE'])(state_below)

        if params['TARGET_TEXT_EMBEDDING_SIZE'] == params['SOURCE_TEXT_EMBEDDING_SIZE']:
            positional_embedding_trg = positional_embedding
        else:
            max_len = max(params['MAX_OUTPUT_TEXT_LEN'], params['MAX_OUTPUT_TEXT_LEN_TEST'])

            positional_embedding_trg = Embedding(max_len,
                                                 params['TARGET_TEXT_EMBEDDING_SIZE'],
                                                 name='positional_trg_word_embedding',
                                                 trainable=False,
                                                 weights=getPositionalEncodingWeights(max_len,
                                                                                      params['TARGET_TEXT_EMBEDDING_SIZE'],
                                                                                      name='positional_trg_word_embedding',
                                                                                      verbose=self.verbose))

        positional_trg_embedding = positional_embedding_trg(next_words_positions)

        state_below = Add()([state_below, positional_trg_embedding])

        # Regularize
        state_below = Dropout(params['DROPOUT_P'])(state_below)

        shared_trg_multihead_list = []
        shared_trg_dropout_multihead_list = []
        shared_trg_add_multihead_list = []
        shared_trg_norm_multihead_list = []

        shared_src_trg_multihead_list = []
        shared_src_trg_dropout_multihead_list = []
        shared_src_trg_add_multihead_list = []
        shared_src_trg_norm_multihead_list = []

        shared_ff_list = []
        shared_dropout_ff_list = []
        shared_add_ff_list = []
        shared_norm_ff_list = []

        prev_state_below = state_below

        # Right tranformer block (decoder)
        for n_block in range(params['N_LAYERS_DECODER']):

            # Declare shared layers of each block

            # Masked Multi-Head Attention block
            shared_trg_multihead = MultiHeadAttention(params['N_HEADS'],
                                                      params['MODEL_SIZE'],
                                                      activation=params.get('MULTIHEAD_ATTENTION_ACTIVATION', 'relu'),
                                                      use_bias=True,
                                                      dropout=params.get('ATTENTION_DROPOUT_P', 0.),
                                                      mask_future=True,  # Avoid attending on future sequences
                                                      name='trg_MultiHeadAttention_' + str(n_block))
            shared_trg_multihead_list.append(shared_trg_multihead)

            # Regularize
            shared_trg_multihead_dropout = Dropout(params['DROPOUT_P'])
            shared_trg_dropout_multihead_list.append(shared_trg_multihead_dropout)

            # Add
            shared_trg_multihead_add = Add(name='trg_Residual_MultiHeadAttention_' + str(n_block))
            shared_trg_add_multihead_list.append(shared_trg_multihead_add)

            # And norm
            shared_trg_multihead_norm = BatchNormalization(mode=1, name='trg_Normalization_MultiHeadAttention_' + str(n_block))
            shared_trg_norm_multihead_list.append(shared_trg_multihead_norm)

            # Second Multi-Head Attention block
            shared_src_trg_multihead = MultiHeadAttention(params['N_HEADS'],
                                                          params['MODEL_SIZE'],
                                                          activation=params.get('MULTIHEAD_ATTENTION_ACTIVATION', 'relu'),
                                                          use_bias=True,
                                                          dropout=params.get('ATTENTION_DROPOUT_P', 0.),
                                                          name='src_trg_MultiHeadAttention_' + str(n_block))
            shared_src_trg_multihead_list.append(shared_src_trg_multihead)

            # Regularize
            shared_src_trg_multihead_dropout = Dropout(params['DROPOUT_P'])
            shared_src_trg_dropout_multihead_list.append(shared_src_trg_multihead_dropout)

            # Add
            shared_src_trg_multihead_add = Add(name='src_trg_Residual_MultiHeadAttention_' + str(n_block))
            shared_src_trg_add_multihead_list.append(shared_src_trg_multihead_add)

            # And norm
            shared_src_trg_multihead_norm = BatchNormalization(mode=1, name='src_trg_Normalization_MultiHeadAttention_' + str(n_block))
            shared_src_trg_norm_multihead_list.append(shared_src_trg_multihead_norm)

            # FF
            shared_ff_src_trg_multihead = TimeDistributed(PositionwiseFeedForwardDense(params['FF_SIZE'],
                                                                                       name='src_trg_PositionwiseFeedForward_' + str(n_block)),
                                                          name='src_trg_TimeDistributedPositionwiseFeedForward_' + str(n_block))
            shared_ff_list.append(shared_ff_src_trg_multihead)

            # Regularize
            shared_ff_src_trg_multihead_dropout = Dropout(params['DROPOUT_P'])
            shared_dropout_ff_list.append(shared_ff_src_trg_multihead_dropout)

            # Add
            shared_ff_src_trg_multihead_add = Add(name='src_trg_Residual_FF_' + str(n_block))
            shared_add_ff_list.append(shared_ff_src_trg_multihead_add)

            # And norm
            shared_ff_src_trg_multihead_norm = BatchNormalization(mode=1, name='src_trg_Normalization_FF_' + str(n_block))
            shared_norm_ff_list.append(shared_ff_src_trg_multihead_norm)

            # Apply shared layers
            # Masked Multi-Head Attention block
            trg_multihead = shared_trg_multihead_list[n_block]([prev_state_below, prev_state_below])

            # Regularize
            trg_multihead_dropout = shared_trg_dropout_multihead_list[n_block](trg_multihead)

            # Add
            trg_multihead_add = shared_trg_add_multihead_list[n_block]([prev_state_below, trg_multihead_dropout])

            # And norm
            trg_multihead_norm = shared_trg_norm_multihead_list[n_block](trg_multihead_add)

            # Second Multi-Head Attention block
            src_trg_multihead = shared_src_trg_multihead_list[n_block]([trg_multihead_norm,   # Queries from the previous decoder layer.
                                                                        masked_src_multihead  # Keys and values from the output of the encoder.
                                                                        ])

            # Regularize
            src_trg_multihead_dropout = shared_src_trg_dropout_multihead_list[n_block](src_trg_multihead)

            # Add
            src_trg_multihead_add = shared_src_trg_add_multihead_list[n_block]([src_trg_multihead_dropout, trg_multihead_norm])

            # And norm
            src_trg_multihead_norm = shared_src_trg_norm_multihead_list[n_block](src_trg_multihead_add)

            # FF
            ff_src_trg_multihead = shared_ff_list[n_block](src_trg_multihead_norm)

            # Regularize
            ff_src_trg_multihead_dropout = shared_dropout_ff_list[n_block](ff_src_trg_multihead)

            # Add
            ff_src_trg_multihead_add = shared_add_ff_list[n_block]([ff_src_trg_multihead_dropout,
                                                                    src_trg_multihead_norm])

            # And norm
            ff_src_trg_multihead_norm = shared_norm_ff_list[n_block](ff_src_trg_multihead_add)

            prev_state_below = ff_src_trg_multihead_norm

        out_layer = prev_state_below
        shared_deep_list = []
        shared_reg_deep_list = []
        # 3.6 Optional deep ouput layer
        for i, (activation, dimension) in enumerate(params['DEEP_OUTPUT_LAYERS']):
            shared_deep_list.append(TimeDistributed(Dense(dimension,
                                                          activation=activation,
                                                          kernel_initializer=params['INIT_FUNCTION'],
                                                          kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                                          bias_regularizer=l2(params['WEIGHT_DECAY']),
                                                          trainable=params.get('TRAINABLE_DECODER', True),
                                                          ),
                                                    trainable=params.get('TRAINABLE_DECODER', True),
                                                    name=activation + '_%d' % i))
            out_layer = shared_deep_list[-1](out_layer)
            [out_layer, shared_reg_out_layer] = Regularize(out_layer,
                                                           params, shared_layers=True,
                                                           name='out_layer_' + str(activation) + '_%d' % i)
            shared_reg_deep_list.append(shared_reg_out_layer)

        # 3.7. Output layer: Softmax
        shared_FC_soft = TimeDistributed(Dense(params['OUTPUT_VOCABULARY_SIZE'],
                                               activation=params['CLASSIFIER_ACTIVATION'],
                                               kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                               bias_regularizer=l2(params['WEIGHT_DECAY']),
                                               trainable=(params.get('TRAINABLE_DECODER', True) or params.get('TRAIN_ONLY_LAST_LAYER', True)),
                                               ),
                                         trainable=(params.get('TRAINABLE_DECODER', True) or params.get('TRAIN_ONLY_LAST_LAYER', True)),
                                         name=self.ids_outputs[0])
        softout = shared_FC_soft(out_layer)
        self.model = Model(inputs=[src_text, next_words],
                           outputs=softout,
                           name=self.name + '_training')

        if params.get('N_GPUS', 1) > 1:
            self.multi_gpu_model = multi_gpu_model(self.model, gpus=params['N_GPUS'])
        else:
            self.multi_gpu_model = None

        ##################################################################
        #                         SAMPLING MODEL                         #
        ##################################################################
        # Now that we have the basic training model ready, let's prepare the model for applying decoding
        # The beam-search model will include all the minimum required set of layers (decoder stage) which offer the
        # possibility to generate the next state in the sequence given a pre-processed input (encoder stage)
        # First, we need a model that outputs the preprocessed input
        # for applying the initial forward pass

        model_init_input = [src_text, next_words]
        model_init_output = [softout, masked_src_multihead]

        self.model_init = Model(inputs=model_init_input,
                                outputs=model_init_output,
                                name=self.name + '_model_init')

        # Store inputs and outputs names for model_init
        self.ids_inputs_init = self.ids_inputs

        # first output must be the output probs.
        self.ids_outputs_init = self.ids_outputs + ['preprocessed_input']

        # Second, we need to build an additional model with the capability to have the following inputs:
        #   - preprocessed_input
        #   - prev_word
        # and the following outputs:
        #   - softmax probabilities

        preprocessed_size = params['MODEL_SIZE']

        # Define inputs
        preprocessed_annotations = Input(name='preprocessed_input',
                                         shape=tuple([None, preprocessed_size]),
                                         dtype='float32')
        # Apply decoder
        prev_state_below = state_below

        # RIGHT TRANSFORMER BLOCK
        for n_block in range(params['N_LAYERS_DECODER']):
            # Masked Multi-Head Attention block
            trg_multihead = shared_trg_multihead_list[n_block]([prev_state_below, prev_state_below])

            # Regularize
            trg_multihead_dropout = shared_trg_dropout_multihead_list[n_block](trg_multihead)

            # Add
            trg_multihead_add = shared_trg_add_multihead_list[n_block]([prev_state_below, trg_multihead_dropout])

            # And norm
            trg_multihead_norm = shared_trg_norm_multihead_list[n_block](trg_multihead_add)

            # Second Multi-Head Attention block
            src_trg_multihead = shared_src_trg_multihead_list[n_block]([trg_multihead_norm,
                                                                        preprocessed_annotations])

            # Regularize
            src_trg_multihead_dropout = shared_src_trg_dropout_multihead_list[n_block](src_trg_multihead)

            # Add
            src_trg_multihead_add = shared_src_trg_add_multihead_list[n_block]([src_trg_multihead_dropout,
                                                                                trg_multihead_norm])

            # And norm
            src_trg_multihead_norm = shared_src_trg_norm_multihead_list[n_block](src_trg_multihead_add)

            # FF
            ff_src_trg_multihead = shared_ff_list[n_block](src_trg_multihead_norm)

            # Regularize
            ff_src_trg_multihead_dropout = shared_dropout_ff_list[n_block](ff_src_trg_multihead)

            # Add
            ff_src_trg_multihead_add = shared_add_ff_list[n_block]([ff_src_trg_multihead_dropout,
                                                                    src_trg_multihead_norm])

            # And norm
            ff_src_trg_multihead_norm = shared_norm_ff_list[n_block](ff_src_trg_multihead_add)

            prev_state_below = ff_src_trg_multihead_norm

        out_layer = prev_state_below

        for (deep_out_layer, reg_list) in zip(shared_deep_list, shared_reg_deep_list):
            out_layer = deep_out_layer(out_layer)
            for reg in reg_list:
                out_layer = reg(out_layer)

        # Softmax
        softout = shared_FC_soft(out_layer)

        model_next_inputs = [next_words, preprocessed_annotations]
        model_next_outputs = [softout, preprocessed_annotations]

        # if self.return_alphas:
        #     model_next_outputs.append(alphas)

        self.model_next = Model(inputs=model_next_inputs,
                                outputs=model_next_outputs,
                                name=self.name + '_model_next')

        # Store inputs and outputs names for model_next
        # first input must be previous word
        self.ids_inputs_next = [self.ids_inputs[1]] + ['preprocessed_input']
        # first output must be the output probs.
        self.ids_outputs_next = self.ids_outputs + ['preprocessed_input']
        # Input -> Output matchings from model_init to model_next and from model_next to model_next
        self.matchings_init_to_next = {'preprocessed_input': 'preprocessed_input'}
        self.matchings_next_to_next = {'preprocessed_input': 'preprocessed_input'}

    # Backwards compatibility.
    GroundHogModel = AttentionRNNEncoderDecoder

    # ----------------------
    #    DeepQuest models
    # ----------------------

    #=============================
    # Word-level QE -- BiRNN model
    #=============================
    #
    ## Inputs:
    # 1. Sentences in src language (shape: (mini_batch_size, line_words))
    # 2. Parallel machine-translated sentences (shape: (mini_batch_size, line_words))
    #
    ## Output:
    # 1. Word quality labels (shape: (mini_batch_size, number_of_qe_labels))
    #
    ## Summary of the model:
    # The sentence-level representations of both the SRC and the MT are created using two bi-directional RNNs.
    # Those representations are then concatenated at the word level, and used for making classification decisions.

    def EncWord(self, params):
        src_words = Input(name=self.ids_inputs[0],
                          batch_shape=tuple([None, params['MAX_INPUT_TEXT_LEN']]), dtype='int32')

        src_embedding = Embedding(params['INPUT_VOCABULARY_SIZE'], params['TARGET_TEXT_EMBEDDING_SIZE'],
                                  name='src_word_embedding',
                                  embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                  embeddings_initializer=params['INIT_FUNCTION'],
                                  trainable=self.trainable,
                                  mask_zero=True)(src_words)
        src_embedding = Regularize(src_embedding, params, trainable=self.trainable, name='src_state')

        src_annotations = Bidirectional(eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                         kernel_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         recurrent_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         bias_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                                         recurrent_dropout=params[
                                                                             'RECURRENT_DROPOUT_P'],
                                                                         kernel_initializer=params['INIT_FUNCTION'],
                                                                         recurrent_initializer=params['INNER_INIT'],
                                                                         return_sequences=True,
                                                                         trainable=self.trainable),
                                        name='src_bidirectional_encoder_' + params['ENCODER_RNN_TYPE'],
                                        merge_mode='concat')(src_embedding)

        trg_words = Input(name=self.ids_inputs[1],
                          batch_shape=tuple([None, params['MAX_INPUT_TEXT_LEN']]), dtype='int32')

        trg_embedding = Embedding(params['OUTPUT_VOCABULARY_SIZE'], params['TARGET_TEXT_EMBEDDING_SIZE'],
                                  name='target_word_embedding',
                                  embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                  embeddings_initializer=params['INIT_FUNCTION'],
                                  trainable=self.trainable,
                                  mask_zero=True)(trg_words)
        trg_embedding = Regularize(trg_embedding, params, trainable=self.trainable, name='state')

        trg_annotations = Bidirectional(eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                     kernel_regularizer=l2(
                                                                         params['RECURRENT_WEIGHT_DECAY']),
                                                                     recurrent_regularizer=l2(
                                                                         params['RECURRENT_WEIGHT_DECAY']),
                                                                     bias_regularizer=l2(
                                                                         params['RECURRENT_WEIGHT_DECAY']),
                                                                     dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                                     recurrent_dropout=params[
                                                                         'RECURRENT_DROPOUT_P'],
                                                                     kernel_initializer=params['INIT_FUNCTION'],
                                                                     recurrent_initializer=params['INNER_INIT'],
                                                                     return_sequences=True, trainable=self.trainable),
                                    name='bidirectional_encoder_' + params['ENCODER_RNN_TYPE'],
                                    merge_mode='concat')(trg_embedding)

        annotations = concatenate([src_annotations, trg_annotations], name='anot_seq_concat')
        out_activation=params.get('OUT_ACTIVATION', 'sigmoid')

        word_qe = TimeDistributed(Dense(params['WORD_QE_CLASSES'], activation=out_activation), name=self.ids_outputs[0])(annotations)

        self.model = Model(inputs=[src_words, trg_words],
                           outputs=[word_qe])



    #=================================
    # Sentence-level QE -- BiRNN model
    #=================================
    #
    ## Inputs:
    # 1. Sentences in src language (shape: (mini_batch_size, line_words))
    # 2. Parallel machine-translated documents (shape: (mini_batch_size, line_words))
    #
    ## Output:
    # 1. Sentence quality scores (shape: (mini_batch_size,))
    #
    ## Summary of the model:
    # The sententence-level representations of both the SRC and the MT are created using two bi-directional RNNs.
    # Those representations are then concatenated at the word level, and the sentence representation is a weighted sum of its words.
    # We apply the following attention function computing a normalized weight for each hidden state of an RNN h_j: 
    #       alpha_j = exp(W_a*h_j)/sum_k exp(W_a*h_k)
    # The resulting sentence vector is thus a weighted sum of word vectors:
    #       v = sum_j alpha_j*h_j
    # Sentence vectors are then directly used for making classification decisions.

    def EncSent(self, params):
        src_words = Input(name=self.ids_inputs[0],
                          batch_shape=tuple([None, params['MAX_INPUT_TEXT_LEN']]), dtype='int32')
        src_embedding = Embedding(params['INPUT_VOCABULARY_SIZE'], params['TARGET_TEXT_EMBEDDING_SIZE'],
                                  name='src_word_embedding',
                                  embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                  embeddings_initializer=params['INIT_FUNCTION'],
                                  trainable=self.trainable,
                                  mask_zero=True)(src_words)
        src_embedding = Regularize(src_embedding, params, trainable=self.trainable, name='src_state')

        src_annotations = Bidirectional(eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                         kernel_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         recurrent_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         bias_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                                         recurrent_dropout=params[
                                                                             'RECURRENT_DROPOUT_P'],
                                                                         kernel_initializer=params['INIT_FUNCTION'],
                                                                         recurrent_initializer=params['INNER_INIT'],
                                                                         return_sequences=True,
                                                                         trainable=self.trainable),
                                        name='src_bidirectional_encoder_' + params['ENCODER_RNN_TYPE'],
                                        merge_mode='concat')(src_embedding)

        trg_words = Input(name=self.ids_inputs[1],
                          batch_shape=tuple([None, params['MAX_INPUT_TEXT_LEN']]), dtype='int32')

        trg_embedding = Embedding(params['OUTPUT_VOCABULARY_SIZE'], params['TARGET_TEXT_EMBEDDING_SIZE'],
                                  name='target_word_embedding',
                                  embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                  embeddings_initializer=params['INIT_FUNCTION'],
                                  trainable=self.trainable,
                                  mask_zero=True)(trg_words)
        trg_embedding = Regularize(trg_embedding, params, trainable=self.trainable, name='state')

        trg_annotations = Bidirectional(eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                     kernel_regularizer=l2(
                                                                         params['RECURRENT_WEIGHT_DECAY']),
                                                                     recurrent_regularizer=l2(
                                                                         params['RECURRENT_WEIGHT_DECAY']),
                                                                     bias_regularizer=l2(
                                                                         params['RECURRENT_WEIGHT_DECAY']),
                                                                     dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                                     recurrent_dropout=params[
                                                                         'RECURRENT_DROPOUT_P'],
                                                                     kernel_initializer=params['INIT_FUNCTION'],
                                                                     recurrent_initializer=params['INNER_INIT'],
                                                                     return_sequences=True, trainable=self.trainable),
                                    name='bidirectional_encoder_' + params['ENCODER_RNN_TYPE'],
                                    merge_mode='concat')(trg_embedding)

        annotations = concatenate([src_annotations, trg_annotations], name='anot_seq_concat')
        annotations = NonMasking()(annotations)
        # apply attention over words at the sentence-level
        annotations = attention_3d_block(annotations, params, 'sent')
        out_activation=params.get('OUT_ACTIVATION', 'sigmoid')
        qe_sent = Dense(1, activation=out_activation, name=self.ids_outputs[0])(annotations)

        self.model = Model(inputs=[src_words, trg_words],
                           outputs=[qe_sent])




    #=================================
    # Document-level QE -- BiRNN model
    #=================================
    #
    ## Inputs:
    # 1. Documents in src language (shape: (mini_batch_size, doc_lines, words))
    # 2. Parallel machine-translated documents (shape: (mini_batch_size, doc_lines, words))
    #
    ## Output:
    # 1. Document quality scores (shape: (mini_batch_size,))
    #
    ## Summary of the model:
    # A hierarchical neural architecture that generalizes sentence-level representations to the document level.
    # The sentence-level representations of both the SRC and the MT are created using two bi-directional RNNs.
    # Those representations are then concatenated at the word level.
    # A sentence representation is a weighted sum of its words.
    # We apply the following attention function computing a normalized weight for each hidden state of an RNN h_j:
    #       alpha_j = exp(W_a*h_j)/sum_k exp(W_a*h_k)
    # The resulting sentence vector is thus a weighted sum of word vectors:
    #       v = sum_j alpha_j*h_j
    # Sentence vectors are inputted in a doc-level bi-directional RNN.
    # The last hidden state of ths RNN is taken as the summary of an entire document.
    # Doc-level representations are then directly used for making classification decisions.
    #
    ## References
    # - Zichao Yang, Diyi Yang, Chris Dyer, Xiaodong He, Alex Smola, and Eduard Hovy. 2016. Hierarchical attention networks for document classification. In Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 1480-1489, San Diego, California, June. Association for Computational Linguistics.
    # - Jiwei Li, Thang Luong, and Dan Jurafsky. 2015. A hierarchical neural autoencoder for paragraphs and docu- ments. In Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 1106-1115, Beijing, China, July. Association for Computational Linguistics.

    def EncDoc(self, params):
        src_words = Input(name=self.ids_inputs[0],
                          batch_shape=tuple([None, params['SECOND_DIM_SIZE'], params['MAX_INPUT_TEXT_LEN']]), dtype='int32')
        # Reshape input to 2d to produce sent-level vectors
        genreshape = GeneralReshape((None, params['MAX_INPUT_TEXT_LEN']), params)
        src_words_in = genreshape(src_words)

        src_embedding = Embedding(params['INPUT_VOCABULARY_SIZE'], params['TARGET_TEXT_EMBEDDING_SIZE'],
                                  name='src_word_embedding',
                                  embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                  embeddings_initializer=params['INIT_FUNCTION'],
                                  trainable=self.trainable,
                                  mask_zero=True)(src_words_in)
        src_embedding = Regularize(src_embedding, params, trainable=self.trainable, name='src_state')

        src_annotations = Bidirectional(eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                         kernel_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         recurrent_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         bias_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                                         recurrent_dropout=params[
                                                                             'RECURRENT_DROPOUT_P'],
                                                                         kernel_initializer=params['INIT_FUNCTION'],
                                                                         recurrent_initializer=params['INNER_INIT'],
                                                                         return_sequences=True,
                                                                         trainable=self.trainable),
                                        name='src_bidirectional_encoder_' + params['ENCODER_RNN_TYPE'],
                                        merge_mode='concat')(src_embedding)

        trg_words = Input(name=self.ids_inputs[1],
                          batch_shape=tuple([None, params['SECOND_DIM_SIZE'], params['MAX_INPUT_TEXT_LEN']]), dtype='int32')
        # Reshape input to 2d to produce sent-level vectors
        trg_words_in = genreshape(trg_words)

        trg_embedding = Embedding(params['OUTPUT_VOCABULARY_SIZE'], params['TARGET_TEXT_EMBEDDING_SIZE'],
                                  name='target_word_embedding',
                                  embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                  embeddings_initializer=params['INIT_FUNCTION'],
                                  trainable=self.trainable,
                                  mask_zero=True)(trg_words_in)
        trg_embedding = Regularize(trg_embedding, params, trainable=self.trainable, name='state')

        trg_annotations = Bidirectional(eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                     kernel_regularizer=l2(
                                                                         params['RECURRENT_WEIGHT_DECAY']),
                                                                     recurrent_regularizer=l2(
                                                                         params['RECURRENT_WEIGHT_DECAY']),
                                                                     bias_regularizer=l2(
                                                                         params['RECURRENT_WEIGHT_DECAY']),
                                                                     dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                                     recurrent_dropout=params[
                                                                         'RECURRENT_DROPOUT_P'],
                                                                     kernel_initializer=params['INIT_FUNCTION'],
                                                                     recurrent_initializer=params['INNER_INIT'],
                                                                     return_sequences=True, trainable=self.trainable),
                                    name='bidirectional_encoder_' + params['ENCODER_RNN_TYPE'],
                                    merge_mode='concat')(trg_embedding)

        annotations = concatenate([src_annotations, trg_annotations], name='anot_seq_concat')
        annotations = NonMasking()(annotations)
        # apply sent-level attention over words
        annotations = attention_3d_block(annotations, params, 'sent')
        # reshape back to 3d input to group sent vectors per doc
        genreshape_out = GeneralReshape((None, params['SECOND_DIM_SIZE'], params['ENCODER_HIDDEN_SIZE'] * 4), params)
        annotations = genreshape_out(annotations)

        # bi-RNN over doc sentences
        dec_doc_frw, dec_doc_last_state_frw = GRU(params['DOC_DECODER_HIDDEN_SIZE'], return_sequences=True, return_state=True,
                                         name='dec_doc_frw')(annotations)
        dec_doc_bkw, dec_doc_last_state_bkw = GRU(params['DOC_DECODER_HIDDEN_SIZE'], return_sequences=True, return_state=True,
                                         go_backwards=True, name='dec_doc_bkw')(annotations)

        dec_doc_bkw = Reverse(dec_doc_bkw._keras_shape[2], axes=1,
                            name='dec_reverse_doc_bkw')(dec_doc_bkw)

        dec_doc_seq_concat = concatenate([dec_doc_frw, dec_doc_bkw], trainable=self.trainable_est, name='dec_doc_seq_concat')

        # we take the last bi-RNN state as doc summary
        dec_doc_last_state_concat = concatenate([dec_doc_last_state_frw, dec_doc_last_state_bkw], name='dec_doc_last_state_concat')
        out_activation=params.get('OUT_ACTIVATION', 'sigmoid')
        qe_doc = Dense(1, activation=out_activation, name=self.ids_outputs[0])(dec_doc_last_state_concat)

        self.model = Model(inputs=[src_words, trg_words],
                           outputs=[qe_doc])



    #=============================================================================
    # Document-level QE with Attention mechanism -- BiRNN model Doc QE + Attention
    #=============================================================================
    #
    ## Inputs:
    # 1. Documents in src language (shape: (mini_batch_size, doc_lines, words))
    # 2. Parallel machine-translated documents (shape: (mini_batch_size, doc_lines, words))
    #
    ## Output:
    # 1. Document quality scores (shape: (mini_batch_size,))
    #
    ## Summary of the model:
    # A hierarchical neural architecture that generalizes sentence-level representations to the      document level.
    # The sentence-level representations of both the SRC and the MT are created using two bi-directional RNNs.
    # Those representations are then concatenated at the word level.
    # A sentence representation is a weighted sum of its words.
    # We apply the following attention function computing a normalized weight for each hidden state of an RNN h_j:
    #       alpha_j = exp(W_a*h_j)/sum_k exp(W_a*h_k)
    # The resulting sentence vector is thus a weighted sum of word vectors:
    #       v = sum_j alpha_j*h_j
    # Sentence vectors are inputted in a doc-level bi-directional RNN.
    # A document representation is a weighted sum of its sentences. We apply the attention function as described above.
    # Doc-level representations are then directly used for making classification decisions.
    #
    ## References
    # - Zichao Yang, Diyi Yang, Chris Dyer, Xiaodong He, Alex Smola, and Eduard Hovy. 2016. Hierarchical attention networks for document classification. In Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 1480-1489, San Diego, California, June. Association for Computational Linguistics.
    # - Jiwei Li, Thang Luong, and Dan Jurafsky. 2015. A hierarchical neural autoencoder for paragraphs and docu- ments. In Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 1106-1115, Beijing, China, July. Association for Computational Linguistics.

    def EncDocAtt(self, params):
        src_words = Input(name=self.ids_inputs[0],
                          batch_shape=tuple([None, params['SECOND_DIM_SIZE'], params['MAX_INPUT_TEXT_LEN']]), dtype='int32')
        # Reshape input to 2d to produce sent-level vectors
        genreshape = GeneralReshape((None, params['MAX_INPUT_TEXT_LEN']), params)
        src_words_in = genreshape(src_words)

        src_embedding = Embedding(params['INPUT_VOCABULARY_SIZE'], params['TARGET_TEXT_EMBEDDING_SIZE'],
                                  name='src_word_embedding',
                                  embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                  embeddings_initializer=params['INIT_FUNCTION'],
                                  trainable=self.trainable,
                                  mask_zero=True)(src_words_in)
        src_embedding = Regularize(src_embedding, params, trainable=self.trainable, name='src_state')

        src_annotations = Bidirectional(eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                         kernel_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         recurrent_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         bias_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                                         recurrent_dropout=params[
                                                                             'RECURRENT_DROPOUT_P'],
                                                                         kernel_initializer=params['INIT_FUNCTION'],
                                                                         recurrent_initializer=params['INNER_INIT'],
                                                                         return_sequences=True,
                                                                         trainable=self.trainable),
                                        name='src_bidirectional_encoder_' + params['ENCODER_RNN_TYPE'],
                                        merge_mode='concat')(src_embedding)

        trg_words = Input(name=self.ids_inputs[1],
                          batch_shape=tuple([None, params['SECOND_DIM_SIZE'], params['MAX_INPUT_TEXT_LEN']]), dtype='int32')
        # Reshape input to 2d to produce sent-level vectors
        trg_words_in = genreshape(trg_words)

        trg_embedding = Embedding(params['OUTPUT_VOCABULARY_SIZE'], params['TARGET_TEXT_EMBEDDING_SIZE'],
                                  name='target_word_embedding',
                                  embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                  embeddings_initializer=params['INIT_FUNCTION'],
                                  trainable=self.trainable,
                                  mask_zero=True)(trg_words_in)
        trg_embedding = Regularize(trg_embedding, params, trainable=self.trainable, name='state')

        trg_annotations = Bidirectional(eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                     kernel_regularizer=l2(
                                                                         params['RECURRENT_WEIGHT_DECAY']),
                                                                     recurrent_regularizer=l2(
                                                                         params['RECURRENT_WEIGHT_DECAY']),
                                                                     bias_regularizer=l2(
                                                                         params['RECURRENT_WEIGHT_DECAY']),
                                                                     dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                                     recurrent_dropout=params[
                                                                         'RECURRENT_DROPOUT_P'],
                                                                     kernel_initializer=params['INIT_FUNCTION'],
                                                                     recurrent_initializer=params['INNER_INIT'],
                                                                     return_sequences=True, trainable=self.trainable),
                                    name='bidirectional_encoder_' + params['ENCODER_RNN_TYPE'],
                                    merge_mode='concat')(trg_embedding)

        annotations = concatenate([src_annotations, trg_annotations], name='anot_seq_concat')
        annotations = NonMasking()(annotations)
        # apply sent-level attention over words
        annotations = attention_3d_block(annotations, params, 'sent')
        # reshape back to 3d input to group sent vectors per doc
        genreshape_out = GeneralReshape((None, params['SECOND_DIM_SIZE'], params['ENCODER_HIDDEN_SIZE'] * 4), params)
        annotations = genreshape_out(annotations)

        #bi-RNN over doc sentences
        dec_doc_frw, dec_doc_last_state_frw = GRU(params['DOC_DECODER_HIDDEN_SIZE'], return_sequences=True, return_state=True,
                                         name='dec_doc_frw')(annotations)
        dec_doc_bkw, dec_doc_last_state_bkw = GRU(params['DOC_DECODER_HIDDEN_SIZE'], return_sequences=True, return_state=True,
                                         go_backwards=True, name='dec_doc_bkw')(annotations)

        dec_doc_bkw = Reverse(dec_doc_bkw._keras_shape[2], axes=1,
                            name='dec_reverse_doc_bkw')(dec_doc_bkw)

        dec_doc_seq_concat = concatenate([dec_doc_frw, dec_doc_bkw], trainable=self.trainable_est, name='dec_doc_seq_concat')

        dec_doc_last_state_concat = concatenate([dec_doc_last_state_frw, dec_doc_last_state_bkw], name='dec_doc_last_state_concat')
        dec_doc_seq_concat = NonMasking()(dec_doc_seq_concat)

        # apply attention over doc sentences
        attention_mul = attention_3d_block(dec_doc_seq_concat, params, 'doc')
        out_activation=params.get('OUT_ACTIVATION', 'sigmoid')

        qe_doc = Dense(1, activation=out_activation, name=self.ids_outputs[0])(attention_mul)

        self.model = Model(inputs=[src_words, trg_words],
                           outputs=[qe_doc])




    #===========================================
    # Document-level QE --POSTECH-inspired model
    #===========================================
    #
    ## Inputs:
    # 1. Documents in src language (shape: (mini_batch_size, doc_lines, words))
    # 2. Machine-translated documents with one-position left-shifted sentences to represent the right context (shape: (mini_batch_size, doc_lines, words))
    # 3. Machine-translated documents with one-position rigth-shifted sentences to represent the left context (shape: (mini_batch_size, doc_lines, words))
    # 4. Machine-translated documents with unshifted sentences for evaluation (shape: (mini_batch_size, doc_lines,words))
    #
    ## Output:
    # 1. Document quality scores (shape: (mini_batch_size,))
    #
    ## Summary of the model:
    # A hierarchical neural architecture that generalizes sentence-level representations to the      document level.
    # Sentence-level representations are created as by a POSTECH-inspired sentence-level QE model.
    # Those representations are inputted in a doc-level bi-directional RNN.
    # The last hidden state of ths RNN is taken as the summary of an entire document.
    # Doc-level representations are then directly used for making classification decisions.
    #
    ## References
    # - Hyun Kim, Hun-Young Jung, Hongseok Kwon, Jong-Hyeok Lee, and Seung-Hoon Na. 2017a. Predictor- estimator: Neural quality estimation based on target word prediction for machine translation. ACM Trans. Asian Low-Resour. Lang. Inf. Process., 17(1):3:1-3:22, September.
    # - Zichao Yang, Diyi Yang, Chris Dyer, Xiaodong He, Alex Smola, and Eduard Hovy. 2016. Hierarchical attention networks for document classification. In Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 1480-1489, San Diego, California, June. Association for Computational Linguistics.
    # - Jiwei Li, Thang Luong, and Dan Jurafsky. 2015. A hierarchical neural autoencoder for paragraphs and docu- ments. In Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 1106-1115, Beijing, China, July. Association for Computational Linguistics.

    def EstimatorDoc(self, params):
        src_text = Input(name=self.ids_inputs[0],
                         batch_shape=tuple([None, params['SECOND_DIM_SIZE'], params['MAX_INPUT_TEXT_LEN']]), dtype='int32')

        # Reshape input to 2d to produce sent-level vectors. Reshaping to (None, None) is necessary for compatibility with pre-trained Predictors.
        genreshape = GeneralReshape((None, None), params)
        src_text_in = genreshape(src_text)
        src_embedding = Embedding(params['INPUT_VOCABULARY_SIZE'], params['SOURCE_TEXT_EMBEDDING_SIZE'],
                                  name='source_word_embedding',
                                  embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                  embeddings_initializer=params['INIT_FUNCTION'],
                                  trainable=self.trainable,
                                  mask_zero=True)(src_text_in)
        src_embedding = Regularize(src_embedding, params, trainable=self.trainable, name='src_embedding')

        if params['BIDIRECTIONAL_ENCODER']:
            annotations = Bidirectional(eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                         kernel_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         recurrent_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         bias_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                                         recurrent_dropout=params[
                                                                             'RECURRENT_DROPOUT_P'],
                                                                         kernel_initializer=params['INIT_FUNCTION'],
                                                                         recurrent_initializer=params['INNER_INIT'],
                                                                         return_sequences=True,
                                                                         trainable=self.trainable),
                                        name='bidirectional_encoder_' + params['ENCODER_RNN_TYPE'],
                                        merge_mode='concat')(src_embedding)
        else:
            annotations = eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                           kernel_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                           recurrent_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                           bias_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                           dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                           recurrent_dropout=params['RECURRENT_DROPOUT_P'],
                                                           kernel_initializer=params['INIT_FUNCTION'],
                                                           recurrent_initializer=params['INNER_INIT'],
                                                           return_sequences=True,
                                                           name='encoder_' + params['ENCODER_RNN_TYPE'],
                                                           trainable=self.trainable)(src_embedding)
        annotations = Regularize(annotations, params, trainable=self.trainable, name='annotations')

        for n_layer in range(1, params['N_LAYERS_ENCODER']):
            if params['BIDIRECTIONAL_DEEP_ENCODER']:
                current_annotations = Bidirectional(eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                                     kernel_regularizer=l2(
                                                                                         params[
                                                                                             'RECURRENT_WEIGHT_DECAY']),
                                                                                     recurrent_regularizer=l2(
                                                                                         params[
                                                                                             'RECURRENT_WEIGHT_DECAY']),
                                                                                     bias_regularizer=l2(
                                                                                         params[
                                                                                             'RECURRENT_WEIGHT_DECAY']),
                                                                                     dropout=params[
                                                                                         'RECURRENT_INPUT_DROPOUT_P'],
                                                                                     recurrent_dropout=params[
                                                                                         'RECURRENT_DROPOUT_P'],
                                                                                     kernel_initializer=params[
                                                                                         'INIT_FUNCTION'],
                                                                                     recurrent_initializer=params[
                                                                                         'INNER_INIT'],
                                                                                     return_sequences=True,
                                                                                     trainable=self.trainable,
                                                                                     ),
                                                    merge_mode='concat',
                                                    name='bidirectional_encoder_' + str(n_layer))(annotations)
            else:
                current_annotations = eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                       kernel_regularizer=l2(
                                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                                       recurrent_regularizer=l2(
                                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                                       bias_regularizer=l2(
                                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                                       dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                                       recurrent_dropout=params['RECURRENT_DROPOUT_P'],
                                                                       kernel_initializer=params['INIT_FUNCTION'],
                                                                       recurrent_initializer=params['INNER_INIT'],
                                                                       return_sequences=True,
                                                                       trainable=self.trainable,
                                                                       name='encoder_' + str(n_layer))(annotations)
            current_annotations = Regularize(current_annotations, params, trainable=self.trainable,
                                             name='annotations_' + str(n_layer))
            annotations = Add(trainable=self.trainable)([annotations, current_annotations])

        # 3. Decoder
        # 3.1.1. Previously generated words as inputs for training -> Teacher forcing
        next_words = Input(name=self.ids_inputs[1],
                           batch_shape=tuple([None, params['SECOND_DIM_SIZE'], params['MAX_INPUT_TEXT_LEN']]), dtype='int32')
        # Reshape input to 2d to produce sent-level vectors
        next_words_in = genreshape(next_words)

        next_words_bkw = Input(name=self.ids_inputs[2],
                               batch_shape=tuple([None, params['SECOND_DIM_SIZE'], params['MAX_INPUT_TEXT_LEN']]),
                               dtype='int32')
        # Reshape input to 2d to produce sent-level vectors
        next_words_bkw_in = genreshape(next_words_bkw)
        # 3.1.2. Target word embedding
        state_below = Embedding(params['OUTPUT_VOCABULARY_SIZE'], params['TARGET_TEXT_EMBEDDING_SIZE'],
                                name='target_word_embedding_below',
                                embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                embeddings_initializer=params['INIT_FUNCTION'],
                                trainable=self.trainable,
                                mask_zero=True)(next_words_in)
        state_below = Regularize(state_below, params, trainable=self.trainable, name='state_below')

        state_above = Embedding(params['OUTPUT_VOCABULARY_SIZE'], params['TARGET_TEXT_EMBEDDING_SIZE'],
                                name='target_word_embedding_above',
                                embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                embeddings_initializer=params['INIT_FUNCTION'],
                                trainable=self.trainable,
                                mask_zero=True)(next_words_bkw_in)
        state_above = Regularize(state_above, params, trainable=self.trainable, name='state_above')

        # 3.2. Decoder's RNN initialization perceptrons with ctx mean
        ctx_mean = MaskedMean(trainable=self.trainable)(annotations)
        annotations = MaskLayer(trainable=self.trainable)(annotations)  # We may want the padded annotations

        if len(params['INIT_LAYERS']) > 0:
            for n_layer_init in range(len(params['INIT_LAYERS']) - 1):
                ctx_mean = Dense(params['DECODER_HIDDEN_SIZE'], name='init_layer_%d' % n_layer_init,
                                 kernel_initializer=params['INIT_FUNCTION'],
                                 kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                 bias_regularizer=l2(params['WEIGHT_DECAY']),
                                 activation=params['INIT_LAYERS'][n_layer_init],
                                 trainable=self.trainable
                                 )(ctx_mean)
                ctx_mean = Regularize(ctx_mean, params, trainable=self.trainable, name='ctx' + str(n_layer_init))

            initial_state = Dense(params['DECODER_HIDDEN_SIZE'], name='initial_state',
                                  kernel_initializer=params['INIT_FUNCTION'],
                                  kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                  bias_regularizer=l2(params['WEIGHT_DECAY']),
                                  activation=params['INIT_LAYERS'][-1],
                                  trainable=self.trainable
                                  )(ctx_mean)
            initial_state = Regularize(initial_state, params, trainable=self.trainable, name='initial_state')
            input_attentional_decoder = [state_below, annotations, initial_state]

            if 'LSTM' in params['DECODER_RNN_TYPE']:
                initial_memory = Dense(params['DECODER_HIDDEN_SIZE'], name='initial_memory',
                                       kernel_initializer=params['INIT_FUNCTION'],
                                       kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                       bias_regularizer=l2(params['WEIGHT_DECAY']),
                                       activation=params['INIT_LAYERS'][-1],
                                       trainable=self.trainable)(ctx_mean)
                initial_memory = Regularize(initial_memory, params, trainable=self.trainable, name='initial_memory')
                input_attentional_decoder.append(initial_memory)
        else:
            # Initialize to zeros vector
            input_attentional_decoder = [state_below, annotations]
            initial_state = ZeroesLayer(params['DECODER_HIDDEN_SIZE'], trainable=self.trainable)(ctx_mean)
            input_attentional_decoder.append(initial_state)
            if 'LSTM' in params['DECODER_RNN_TYPE']:
                input_attentional_decoder.append(initial_state)

        # 3.3. Attentional decoder
        sharedAttRNNCond = eval('Att' + params['DECODER_RNN_TYPE'] + 'Cond')(params['DECODER_HIDDEN_SIZE'],
                                                                             att_units=params.get('ATTENTION_SIZE', 0),
                                                                             kernel_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             recurrent_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             conditional_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             bias_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             attention_context_wa_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             attention_recurrent_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             attention_context_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             bias_ba_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             dropout=params[
                                                                                 'RECURRENT_INPUT_DROPOUT_P'],
                                                                             recurrent_dropout=params[
                                                                                 'RECURRENT_DROPOUT_P'],
                                                                             conditional_dropout=params[
                                                                                 'RECURRENT_INPUT_DROPOUT_P'],
                                                                             attention_dropout=params['DROPOUT_P'],
                                                                             kernel_initializer=params['INIT_FUNCTION'],
                                                                             recurrent_initializer=params['INNER_INIT'],
                                                                             attention_context_initializer=params[
                                                                                 'INIT_ATT'],
                                                                             return_sequences=True,
                                                                             return_extra_variables=True,
                                                                             return_states=True,
                                                                             num_inputs=len(input_attentional_decoder),
                                                                             name='decoder_Att' + params[
                                                                                 'DECODER_RNN_TYPE'] + 'Cond',
                                                                             trainable=self.trainable)

        rnn_output = sharedAttRNNCond(input_attentional_decoder)
        proj_h = rnn_output[0]
        x_att = rnn_output[1]
        alphas = rnn_output[2]
        h_state = rnn_output[3]

        trg_enc_frw = eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                       kernel_regularizer=l2(
                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                       recurrent_regularizer=l2(
                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                       bias_regularizer=l2(
                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                       dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                       recurrent_dropout=params[
                                                           'RECURRENT_DROPOUT_P'],
                                                       kernel_initializer=params['INIT_FUNCTION'],
                                                       recurrent_initializer=params['INNER_INIT'],
                                                       return_sequences=True,
                                                       trainable=self.trainable,
                                                       name='enc_trg_frw')

        trg_enc_bkw = eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                       kernel_regularizer=l2(
                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                       recurrent_regularizer=l2(
                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                       bias_regularizer=l2(
                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                       dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                       recurrent_dropout=params[
                                                           'RECURRENT_DROPOUT_P'],
                                                       kernel_initializer=params['INIT_FUNCTION'],
                                                       recurrent_initializer=params['INNER_INIT'],
                                                       return_sequences=True,
                                                       trainable=self.trainable,
                                                       go_backwards=True,
                                                       name='enc_trg_bkw')

        trg_state_frw = trg_enc_frw(state_below)
        trg_state_bkw = trg_enc_bkw(state_above)

        trg_state_bkw = Reverse(trg_state_bkw._keras_shape[2], axes=1, trainable=self.trainable,
                                name='reverse_trg_state_bkw')(trg_state_bkw)

        # preparing formula 3b
        merged_emb = concatenate([state_below, state_above], axis=2, trainable=self.trainable, name='merged_emb')
        merged_states = concatenate([trg_state_frw, trg_state_bkw], axis=2, trainable=self.trainable,
                                    name='merged_states')

        # we replace state before with the concatenation of state before and after
        proj_h = merged_states

        if 'LSTM' in params['DECODER_RNN_TYPE']:
            h_memory = rnn_output[4]
        shared_Lambda_Permute = PermuteGeneral((1, 0, 2), trainable=self.trainable)

        if params['DOUBLE_STOCHASTIC_ATTENTION_REG'] > 0:
            alpha_regularizer = AlphaRegularizer(alpha_factor=params['DOUBLE_STOCHASTIC_ATTENTION_REG'])(alphas)

        [proj_h, shared_reg_proj_h] = Regularize(proj_h, params, trainable=self.trainable, shared_layers=True,
                                                 name='proj_h0')

        # 3.4. Possibly deep decoder
        shared_proj_h_list = []
        shared_reg_proj_h_list = []

        h_states_list = [h_state]
        if 'LSTM' in params['DECODER_RNN_TYPE']:
            h_memories_list = [h_memory]

        for n_layer in range(1, params['N_LAYERS_DECODER']):
            current_rnn_input = [merged_states, shared_Lambda_Permute(x_att), initial_state]
            shared_proj_h_list.append(eval(params['DECODER_RNN_TYPE'].replace('Conditional', '') + 'Cond')(
                params['DECODER_HIDDEN_SIZE'],
                kernel_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                recurrent_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                conditional_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                bias_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                dropout=params['RECURRENT_DROPOUT_P'],
                recurrent_dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                conditional_dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                kernel_initializer=params['INIT_FUNCTION'],
                recurrent_initializer=params['INNER_INIT'],
                return_sequences=True,
                return_states=True,
                num_inputs=len(current_rnn_input),
                trainable=self.trainable,
                name='decoder_' + params['DECODER_RNN_TYPE'].replace(
                    'Conditional', '') + 'Cond' + str(n_layer)))

            if 'LSTM' in params['DECODER_RNN_TYPE']:
                current_rnn_input.append(initial_memory)
            current_rnn_output = shared_proj_h_list[-1](current_rnn_input)
            current_proj_h = current_rnn_output[0]
            h_states_list.append(current_rnn_output[1])
            if 'LSTM' in params['DECODER_RNN_TYPE']:
                h_memories_list.append(current_rnn_output[2])
            [current_proj_h, shared_reg_proj_h] = Regularize(current_proj_h, params, trainable=self.trainable,
                                                             shared_layers=True,
                                                             name='proj_h' + str(n_layer))
            shared_reg_proj_h_list.append(shared_reg_proj_h)

            proj_h = Add(trainable=self.trainable)([proj_h, current_proj_h])

        # 3.5. Skip connections between encoder and output layer
        shared_FC_mlp = TimeDistributed(Dense(params['SKIP_VECTORS_HIDDEN_SIZE'],
                                              kernel_initializer=params['INIT_FUNCTION'],
                                              kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                              bias_regularizer=l2(params['WEIGHT_DECAY']),
                                              activation='linear',
                                              trainable=self.trainable),
                                        name='logit_lstm')
        out_layer_mlp = shared_FC_mlp(proj_h)
        shared_FC_ctx = TimeDistributed(Dense(params['SKIP_VECTORS_HIDDEN_SIZE'],
                                              kernel_initializer=params['INIT_FUNCTION'],
                                              kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                              bias_regularizer=l2(params['WEIGHT_DECAY']),
                                              activation='linear',
                                              trainable=self.trainable),
                                        name='logit_ctx')
        out_layer_ctx = shared_FC_ctx(x_att)
        out_layer_ctx = shared_Lambda_Permute(out_layer_ctx)
        shared_FC_emb = TimeDistributed(Dense(params['SKIP_VECTORS_HIDDEN_SIZE'],
                                              kernel_initializer=params['INIT_FUNCTION'],
                                              kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                              bias_regularizer=l2(params['WEIGHT_DECAY']),
                                              activation='linear',
                                              trainable=self.trainable),
                                        name='logit_emb')
        out_layer_emb = shared_FC_emb(merged_emb)

        [out_layer_mlp, shared_reg_out_layer_mlp] = Regularize(out_layer_mlp, params,
                                                               shared_layers=True, trainable=self.trainable,
                                                               name='out_layer_mlp')
        [out_layer_ctx, shared_reg_out_layer_ctx] = Regularize(out_layer_ctx, params,
                                                               shared_layers=True, trainable=self.trainable,
                                                               name='out_layer_ctx')
        [out_layer_emb, shared_reg_out_layer_emb] = Regularize(out_layer_emb, params,
                                                               shared_layers=True, trainable=self.trainable,
                                                               name='out_layer_emb')

        shared_additional_output_merge = eval(params['ADDITIONAL_OUTPUT_MERGE_MODE'])(name='additional_input',
                                                                                      trainable=self.trainable)

        # formula 3b addition
        additional_output = shared_additional_output_merge([out_layer_mlp, out_layer_ctx, out_layer_emb])
        shared_activation_tanh = Activation('tanh', trainable=self.trainable)

        out_layer = shared_activation_tanh(additional_output)

        shared_deep_list = []
        shared_reg_deep_list = []

        # 3.6 Optional deep ouput layer
        for i, (activation, dimension) in enumerate(params['DEEP_OUTPUT_LAYERS']):
            shared_deep_list.append(TimeDistributed(Dense(dimension, activation=activation,
                                                          kernel_initializer=params['INIT_FUNCTION'],
                                                          kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                                          bias_regularizer=l2(params['WEIGHT_DECAY']),
                                                          trainable=self.trainable),
                                                    name=activation + '_%d' % i))
            out_layer = shared_deep_list[-1](out_layer)
            [out_layer, shared_reg_out_layer] = Regularize(out_layer,
                                                           params, trainable=self.trainable, shared_layers=True,
                                                           name='out_layer_' + str(activation) + '_%d' % i)
            shared_reg_deep_list.append(shared_reg_out_layer)

        shared_QE_soft = TimeDistributed(Dense(params['QE_VECTOR_SIZE'],
                                               use_bias=False,
                                               kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                               bias_regularizer=l2(params['WEIGHT_DECAY']),
                                               name='QE' + params['CLASSIFIER_ACTIVATION'],
                                               trainable=self.trainable),
                                         name='QE' + 'target_text')

        # 3.7. Output layer: Softmax
        shared_FC_soft = TimeDistributed(Dense(params['OUTPUT_VOCABULARY_SIZE'],
                                               use_bias=False,
                                               activation=params['CLASSIFIER_ACTIVATION'],
                                               kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                               bias_regularizer=l2(params['WEIGHT_DECAY']),
                                               name=params['CLASSIFIER_ACTIVATION'],
                                               trainable=self.trainable),
                                         name='target-text')

        softoutQE = shared_QE_soft(out_layer)
        softout = shared_FC_soft(softoutQE)

        trg_words = Input(name=self.ids_inputs[3],
                          batch_shape=tuple([None, params['SECOND_DIM_SIZE'], params['MAX_INPUT_TEXT_LEN']]), dtype='int32')
        trg_words_in = genreshape(trg_words)

        next_words_one_hot = one_hot(trg_words_in, params)

        # we multiply the weight matrix by one-hot reference vector to make vectors for the words we don't need zeroes
        qv_prep = DenseTranspose(params['QE_VECTOR_SIZE'], shared_FC_soft, self.ids_outputs[0], name='transpose',
                                 trainable=self.trainable_est)

        qv_prep_res = qv_prep(next_words_one_hot)

        # we multiply our matrix with zero values by our quality vectors so that zero vectors do not influence our decisions
        qv = multiply([qv_prep_res, softoutQE], name='qv')

        enc_qe_frw, last_state_frw = GRU(params['QE_VECTOR_SIZE'], return_sequences=True, return_state=True,
                                         name='qe_frw', trainable=self.trainable_est)(qv)
        enc_qe_bkw, last_state_bkw = GRU(params['QE_VECTOR_SIZE'], return_sequences=True, return_state=True,
                                         go_backwards=True, name='qe_bkw', trainable=self.trainable_est)(qv)

        enc_qe_bkw = Reverse(enc_qe_bkw._keras_shape[2], axes=1, trainable=self.trainable_est,
                             name='reverse_enc_qe_bkw')(enc_qe_bkw)

        enc_qe_concat = concatenate([enc_qe_frw, enc_qe_bkw], name='enc_qe_concat')
        last_state_concat = concatenate([last_state_frw, last_state_bkw], trainable=self.trainable_est,
                                        name='last_state_concat')

        #reshape back to 3d input to group sent vectors per doc
        genreshape_out = GeneralReshape((None, params['SECOND_DIM_SIZE'], params['QE_VECTOR_SIZE'] * 2), params)

        last_state_concat = genreshape_out(last_state_concat)

        #bi-RNN over doc sentences
        dec_doc_frw, dec_doc_last_state_frw = GRU(params['DOC_DECODER_HIDDEN_SIZE'], return_sequences=True,
                                                  return_state=True,
                                                  name='dec_doc_frw')(last_state_concat)
        dec_doc_bkw, dec_doc_last_state_bkw = GRU(params['DOC_DECODER_HIDDEN_SIZE'], return_sequences=True,
                                                  return_state=True,
                                                  go_backwards=True, name='dec_doc_bkw')(last_state_concat)

        dec_doc_bkw = Reverse(dec_doc_bkw._keras_shape[2], axes=1,
                              name='dec_reverse_doc_bkw')(dec_doc_bkw)

        dec_doc_seq_concat = concatenate([dec_doc_frw, dec_doc_bkw], trainable=self.trainable_est,
                                         name='dec_doc_seq_concat')

        #we take the last bi-RNN state as doc summary
        dec_doc_last_state_concat = concatenate([dec_doc_last_state_frw, dec_doc_last_state_bkw],
                                                name='dec_doc_last_state_concat')
        out_activation=params.get('OUT_ACTIVATION', 'sigmoid')
        qe_doc = Dense(1, activation=out_activation, name=self.ids_outputs[0])(dec_doc_last_state_concat)
        self.model = Model(inputs=[src_text, next_words, next_words_bkw, trg_words],
                           outputs=[qe_doc])





    #=====================================================================
    # Document-level QE with Attention mechanism -- POSTECH-inspired model
    #=====================================================================
    #
    ## Inputs:
    # 1. Documents in src language (shape: (mini_batch_size, doc_lines, words))
    # 2. Machine-translated documents with one-position left-shifted sentences to represent the right context (shape: (mini_batch_size, doc_lines, words))
    # 3. Machine-translated documents with one-position rigth-shifted sentences to represent the left context (shape: (mini_batch_size, doc_lines, words))
    # 4. Machine-translated documents with unshifted sentences for evaluation (shape: (mini_batch_size, doc_lines, doc_lines,words))
    #
    ## Output:
    # 1. Document quality scores (shape: (mini_batch_size,))
    #
    ## Summary of the model:
    # A hierarchical neural architecture that generalizes sentence-level representations to the      document level.
    # Sentence-level representations are created as by a POSTECH-inspired sentence-level QE model.
    # Those representations are inputted in a doc-level bi-directional RNN.
    # A document representation is a weighted sum of its sentences.
    # We apply the following attention function computing a normalized weight for each hidden state of an RNN h_j: alpha_j = exp(W_a*h_j)/sum_k exp(W_a*h_k)
    # The resulting document vector is thus a weighted sum of sentence vectors:
    # v = sum_j alpha_j*h_j
    # Doc-level representations are then directly used for making classification decisions.
    #
    ## References
    # - Hyun Kim, Hun-Young Jung, Hongseok Kwon, Jong-Hyeok Lee, and Seung-Hoon Na. 2017a. Predictor- estimator: Neural quality estimation based on target word prediction for machine translation. ACM Trans. Asian Low-Resour. Lang. Inf. Process., 17(1):3:1-3:22, September.
    # - Zichao Yang, Diyi Yang, Chris Dyer, Xiaodong He, Alex Smola, and Eduard Hovy. 2016. Hierarchical attention networks for document classification. In Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 1480-1489, San Diego, California, June. Association for Computational Linguistics.
    # - Jiwei Li, Thang Luong, and Dan Jurafsky. 2015. A hierarchical neural autoencoder for paragraphs and docu- ments. In Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 1106-1115, Beijing, China, July. Association for Computational Linguistics.

    def EstimatorDocAtt(self, params):
        src_text = Input(name=self.ids_inputs[0],
                         batch_shape=tuple([None, params['SECOND_DIM_SIZE'], params['MAX_INPUT_TEXT_LEN']]), dtype='int32')

        # Reshape input to 2d to produce sent-level vectors. Reshaping to (None, None) is necessary for compatibility with pre-trained Predictors
        genreshape = GeneralReshape((None, None), params)
        src_text_in = genreshape(src_text)

        src_embedding = Embedding(params['INPUT_VOCABULARY_SIZE'], params['SOURCE_TEXT_EMBEDDING_SIZE'],
                                  name='source_word_embedding',
                                  embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                  embeddings_initializer=params['INIT_FUNCTION'],
                                  trainable=self.trainable,
                                  mask_zero=True)(src_text_in)
        src_embedding = Regularize(src_embedding, params, trainable=self.trainable, name='src_embedding')

        if params['BIDIRECTIONAL_ENCODER']:
            annotations = Bidirectional(eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                         kernel_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         recurrent_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         bias_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                                         recurrent_dropout=params[
                                                                             'RECURRENT_DROPOUT_P'],
                                                                         kernel_initializer=params['INIT_FUNCTION'],
                                                                         recurrent_initializer=params['INNER_INIT'],
                                                                         return_sequences=True,
                                                                         trainable=self.trainable),
                                        name='bidirectional_encoder_' + params['ENCODER_RNN_TYPE'],
                                        merge_mode='concat')(src_embedding)
        else:
            annotations = eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                           kernel_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                           recurrent_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                           bias_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                           dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                           recurrent_dropout=params['RECURRENT_DROPOUT_P'],
                                                           kernel_initializer=params['INIT_FUNCTION'],
                                                           recurrent_initializer=params['INNER_INIT'],
                                                           return_sequences=True,
                                                           name='encoder_' + params['ENCODER_RNN_TYPE'],
                                                           trainable=self.trainable)(src_embedding)
        annotations = Regularize(annotations, params, trainable=self.trainable, name='annotations')
        # 2.3. Potentially deep encoder
        for n_layer in range(1, params['N_LAYERS_ENCODER']):
            if params['BIDIRECTIONAL_DEEP_ENCODER']:
                current_annotations = Bidirectional(eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                                     kernel_regularizer=l2(
                                                                                         params[
                                                                                             'RECURRENT_WEIGHT_DECAY']),
                                                                                     recurrent_regularizer=l2(
                                                                                         params[
                                                                                             'RECURRENT_WEIGHT_DECAY']),
                                                                                     bias_regularizer=l2(
                                                                                         params[
                                                                                             'RECURRENT_WEIGHT_DECAY']),
                                                                                     dropout=params[
                                                                                         'RECURRENT_INPUT_DROPOUT_P'],
                                                                                     recurrent_dropout=params[
                                                                                         'RECURRENT_DROPOUT_P'],
                                                                                     kernel_initializer=params[
                                                                                         'INIT_FUNCTION'],
                                                                                     recurrent_initializer=params[
                                                                                         'INNER_INIT'],
                                                                                     return_sequences=True,
                                                                                     trainable=self.trainable,
                                                                                     ),
                                                    merge_mode='concat',
                                                    name='bidirectional_encoder_' + str(n_layer))(annotations)
            else:
                current_annotations = eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                       kernel_regularizer=l2(
                                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                                       recurrent_regularizer=l2(
                                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                                       bias_regularizer=l2(
                                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                                       dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                                       recurrent_dropout=params['RECURRENT_DROPOUT_P'],
                                                                       kernel_initializer=params['INIT_FUNCTION'],
                                                                       recurrent_initializer=params['INNER_INIT'],
                                                                       return_sequences=True,
                                                                       trainable=self.trainable,
                                                                       name='encoder_' + str(n_layer))(annotations)
            current_annotations = Regularize(current_annotations, params, trainable=self.trainable,
                                             name='annotations_' + str(n_layer))
            annotations = Add(trainable=self.trainable)([annotations, current_annotations])

        # 3. Decoder
        # 3.1.1. Previously generated words as inputs for training -> Teacher forcing
        # trg_words = Input(name=self.ids_inputs[1], batch_shape=tuple([None, None]), dtype='int32')
        next_words = Input(name=self.ids_inputs[1],
                           batch_shape=tuple([None, params['SECOND_DIM_SIZE'], params['MAX_INPUT_TEXT_LEN']]), dtype='int32')
        next_words_in = genreshape(next_words)

        next_words_bkw = Input(name=self.ids_inputs[2],
                               batch_shape=tuple([None, params['SECOND_DIM_SIZE'], params['MAX_INPUT_TEXT_LEN']]),
                               dtype='int32')
        next_words_bkw_in = genreshape(next_words_bkw)

        # 3.1.2. Target word embedding
        state_below = Embedding(params['OUTPUT_VOCABULARY_SIZE'], params['TARGET_TEXT_EMBEDDING_SIZE'],
                                name='target_word_embedding_below',
                                embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                embeddings_initializer=params['INIT_FUNCTION'],
                                trainable=self.trainable,
                                mask_zero=True)(next_words_in)
        state_below = Regularize(state_below, params, trainable=self.trainable, name='state_below')

        state_above = Embedding(params['OUTPUT_VOCABULARY_SIZE'], params['TARGET_TEXT_EMBEDDING_SIZE'],
                                name='target_word_embedding_above',
                                embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                embeddings_initializer=params['INIT_FUNCTION'],
                                trainable=self.trainable,
                                mask_zero=True)(next_words_bkw_in)
        state_above = Regularize(state_above, params, trainable=self.trainable, name='state_above')

        # 3.2. Decoder's RNN initialization perceptrons with ctx mean
        ctx_mean = MaskedMean(trainable=self.trainable)(annotations)
        annotations = MaskLayer(trainable=self.trainable)(annotations)  # We may want the padded annotations

        if len(params['INIT_LAYERS']) > 0:
            for n_layer_init in range(len(params['INIT_LAYERS']) - 1):
                ctx_mean = Dense(params['DECODER_HIDDEN_SIZE'], name='init_layer_%d' % n_layer_init,
                                 kernel_initializer=params['INIT_FUNCTION'],
                                 kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                 bias_regularizer=l2(params['WEIGHT_DECAY']),
                                 activation=params['INIT_LAYERS'][n_layer_init],
                                 trainable=self.trainable
                                 )(ctx_mean)
                ctx_mean = Regularize(ctx_mean, params, trainable=self.trainable, name='ctx' + str(n_layer_init))

            initial_state = Dense(params['DECODER_HIDDEN_SIZE'], name='initial_state',
                                  kernel_initializer=params['INIT_FUNCTION'],
                                  kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                  bias_regularizer=l2(params['WEIGHT_DECAY']),
                                  activation=params['INIT_LAYERS'][-1],
                                  trainable=self.trainable
                                  )(ctx_mean)
            initial_state = Regularize(initial_state, params, trainable=self.trainable, name='initial_state')
            input_attentional_decoder = [state_below, annotations, initial_state]

            if 'LSTM' in params['DECODER_RNN_TYPE']:
                initial_memory = Dense(params['DECODER_HIDDEN_SIZE'], name='initial_memory',
                                       kernel_initializer=params['INIT_FUNCTION'],
                                       kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                       bias_regularizer=l2(params['WEIGHT_DECAY']),
                                       activation=params['INIT_LAYERS'][-1],
                                       trainable=self.trainable)(ctx_mean)
                initial_memory = Regularize(initial_memory, params, trainable=self.trainable, name='initial_memory')
                input_attentional_decoder.append(initial_memory)
        else:
            # Initialize to zeros vector
            input_attentional_decoder = [state_below, annotations]
            initial_state = ZeroesLayer(params['DECODER_HIDDEN_SIZE'], trainable=self.trainable)(ctx_mean)
            input_attentional_decoder.append(initial_state)
            if 'LSTM' in params['DECODER_RNN_TYPE']:
                input_attentional_decoder.append(initial_state)

        # 3.3. Attentional decoder
        sharedAttRNNCond = eval('Att' + params['DECODER_RNN_TYPE'] + 'Cond')(params['DECODER_HIDDEN_SIZE'],
                                                                             att_units=params.get('ATTENTION_SIZE', 0),
                                                                             kernel_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             recurrent_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             conditional_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             bias_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             attention_context_wa_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             attention_recurrent_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             attention_context_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             bias_ba_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             dropout=params[
                                                                                 'RECURRENT_INPUT_DROPOUT_P'],
                                                                             recurrent_dropout=params[
                                                                                 'RECURRENT_DROPOUT_P'],
                                                                             conditional_dropout=params[
                                                                                 'RECURRENT_INPUT_DROPOUT_P'],
                                                                             attention_dropout=params['DROPOUT_P'],
                                                                             kernel_initializer=params['INIT_FUNCTION'],
                                                                             recurrent_initializer=params['INNER_INIT'],
                                                                             attention_context_initializer=params[
                                                                                 'INIT_ATT'],
                                                                             return_sequences=True,
                                                                             return_extra_variables=True,
                                                                             return_states=True,
                                                                             num_inputs=len(input_attentional_decoder),
                                                                             name='decoder_Att' + params[
                                                                                 'DECODER_RNN_TYPE'] + 'Cond',
                                                                             trainable=self.trainable)

        rnn_output = sharedAttRNNCond(input_attentional_decoder)
        proj_h = rnn_output[0]
        x_att = rnn_output[1]
        alphas = rnn_output[2]
        h_state = rnn_output[3]

        trg_enc_frw = eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                       kernel_regularizer=l2(
                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                       recurrent_regularizer=l2(
                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                       bias_regularizer=l2(
                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                       dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                       recurrent_dropout=params[
                                                           'RECURRENT_DROPOUT_P'],
                                                       kernel_initializer=params['INIT_FUNCTION'],
                                                       recurrent_initializer=params['INNER_INIT'],
                                                       return_sequences=True,
                                                       trainable=self.trainable,
                                                       name='enc_trg_frw')

        trg_enc_bkw = eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                       kernel_regularizer=l2(
                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                       recurrent_regularizer=l2(
                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                       bias_regularizer=l2(
                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                       dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                       recurrent_dropout=params[
                                                           'RECURRENT_DROPOUT_P'],
                                                       kernel_initializer=params['INIT_FUNCTION'],
                                                       recurrent_initializer=params['INNER_INIT'],
                                                       return_sequences=True,
                                                       trainable=self.trainable,
                                                       go_backwards=True,
                                                       name='enc_trg_bkw')

        trg_state_frw = trg_enc_frw(state_below)
        trg_state_bkw = trg_enc_bkw(state_above)

        trg_state_bkw = Reverse(trg_state_bkw._keras_shape[2], axes=1, trainable=self.trainable,
                                name='reverse_trg_state_bkw')(trg_state_bkw)

        # preparing formula 3b
        merged_emb = concatenate([state_below, state_above], axis=2, trainable=self.trainable, name='merged_emb')
        merged_states = concatenate([trg_state_frw, trg_state_bkw], axis=2, trainable=self.trainable,
                                    name='merged_states')

        # we replace state before with the concatenation of state before and after
        proj_h = merged_states

        if 'LSTM' in params['DECODER_RNN_TYPE']:
            h_memory = rnn_output[4]
        shared_Lambda_Permute = PermuteGeneral((1, 0, 2), trainable=self.trainable)

        if params['DOUBLE_STOCHASTIC_ATTENTION_REG'] > 0:
            alpha_regularizer = AlphaRegularizer(alpha_factor=params['DOUBLE_STOCHASTIC_ATTENTION_REG'])(alphas)

        [proj_h, shared_reg_proj_h] = Regularize(proj_h, params, trainable=self.trainable, shared_layers=True,
                                                 name='proj_h0')

        # 3.4. Possibly deep decoder
        shared_proj_h_list = []
        shared_reg_proj_h_list = []

        h_states_list = [h_state]
        if 'LSTM' in params['DECODER_RNN_TYPE']:
            h_memories_list = [h_memory]

        for n_layer in range(1, params['N_LAYERS_DECODER']):
            current_rnn_input = [merged_states, shared_Lambda_Permute(x_att), initial_state]
            shared_proj_h_list.append(eval(params['DECODER_RNN_TYPE'].replace('Conditional', '') + 'Cond')(
                params['DECODER_HIDDEN_SIZE'],
                kernel_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                recurrent_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                conditional_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                bias_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                dropout=params['RECURRENT_DROPOUT_P'],
                recurrent_dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                conditional_dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                kernel_initializer=params['INIT_FUNCTION'],
                recurrent_initializer=params['INNER_INIT'],
                return_sequences=True,
                return_states=True,
                num_inputs=len(current_rnn_input),
                trainable=self.trainable,
                name='decoder_' + params['DECODER_RNN_TYPE'].replace(
                    'Conditional', '') + 'Cond' + str(n_layer)))

            if 'LSTM' in params['DECODER_RNN_TYPE']:
                current_rnn_input.append(initial_memory)
            current_rnn_output = shared_proj_h_list[-1](current_rnn_input)
            current_proj_h = current_rnn_output[0]
            h_states_list.append(current_rnn_output[1])
            if 'LSTM' in params['DECODER_RNN_TYPE']:
                h_memories_list.append(current_rnn_output[2])
            [current_proj_h, shared_reg_proj_h] = Regularize(current_proj_h, params, trainable=self.trainable,
                                                             shared_layers=True,
                                                             name='proj_h' + str(n_layer))
            shared_reg_proj_h_list.append(shared_reg_proj_h)

            proj_h = Add(trainable=self.trainable)([proj_h, current_proj_h])

        # 3.5. Skip connections between encoder and output layer
        shared_FC_mlp = TimeDistributed(Dense(params['SKIP_VECTORS_HIDDEN_SIZE'],
                                              kernel_initializer=params['INIT_FUNCTION'],
                                              kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                              bias_regularizer=l2(params['WEIGHT_DECAY']),
                                              activation='linear',
                                              trainable=self.trainable),
                                        name='logit_lstm')
        out_layer_mlp = shared_FC_mlp(proj_h)
        shared_FC_ctx = TimeDistributed(Dense(params['SKIP_VECTORS_HIDDEN_SIZE'],
                                              kernel_initializer=params['INIT_FUNCTION'],
                                              kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                              bias_regularizer=l2(params['WEIGHT_DECAY']),
                                              activation='linear',
                                              trainable=self.trainable),
                                        name='logit_ctx')
        out_layer_ctx = shared_FC_ctx(x_att)
        out_layer_ctx = shared_Lambda_Permute(out_layer_ctx)
        shared_FC_emb = TimeDistributed(Dense(params['SKIP_VECTORS_HIDDEN_SIZE'],
                                              kernel_initializer=params['INIT_FUNCTION'],
                                              kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                              bias_regularizer=l2(params['WEIGHT_DECAY']),
                                              activation='linear',
                                              trainable=self.trainable),
                                        name='logit_emb')
        out_layer_emb = shared_FC_emb(merged_emb)

        [out_layer_mlp, shared_reg_out_layer_mlp] = Regularize(out_layer_mlp, params,
                                                               shared_layers=True, trainable=self.trainable,
                                                               name='out_layer_mlp')
        [out_layer_ctx, shared_reg_out_layer_ctx] = Regularize(out_layer_ctx, params,
                                                               shared_layers=True, trainable=self.trainable,
                                                               name='out_layer_ctx')
        [out_layer_emb, shared_reg_out_layer_emb] = Regularize(out_layer_emb, params,
                                                               shared_layers=True, trainable=self.trainable,
                                                               name='out_layer_emb')

        shared_additional_output_merge = eval(params['ADDITIONAL_OUTPUT_MERGE_MODE'])(name='additional_input',
                                                                                      trainable=self.trainable)

        # formula 3b addition
        additional_output = shared_additional_output_merge([out_layer_mlp, out_layer_ctx, out_layer_emb])
        shared_activation_tanh = Activation('tanh', trainable=self.trainable)

        out_layer = shared_activation_tanh(additional_output)

        shared_deep_list = []
        shared_reg_deep_list = []

        # 3.6 Optional deep ouput layer
        for i, (activation, dimension) in enumerate(params['DEEP_OUTPUT_LAYERS']):
            shared_deep_list.append(TimeDistributed(Dense(dimension, activation=activation,
                                                          kernel_initializer=params['INIT_FUNCTION'],
                                                          kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                                          bias_regularizer=l2(params['WEIGHT_DECAY']),
                                                          trainable=self.trainable),
                                                    name=activation + '_%d' % i))
            out_layer = shared_deep_list[-1](out_layer)
            [out_layer, shared_reg_out_layer] = Regularize(out_layer,
                                                           params, trainable=self.trainable, shared_layers=True,
                                                           name='out_layer_' + str(activation) + '_%d' % i)
            shared_reg_deep_list.append(shared_reg_out_layer)

        shared_QE_soft = TimeDistributed(Dense(params['QE_VECTOR_SIZE'],
                                               use_bias=False,
                                               kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                               bias_regularizer=l2(params['WEIGHT_DECAY']),
                                               name='QE' + params['CLASSIFIER_ACTIVATION'],
                                               trainable=self.trainable),
                                         name='QE' + 'target_text')

        # 3.7. Output layer: Softmax
        shared_FC_soft = TimeDistributed(Dense(params['OUTPUT_VOCABULARY_SIZE'],
                                               use_bias=False,
                                               activation=params['CLASSIFIER_ACTIVATION'],
                                               kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                               bias_regularizer=l2(params['WEIGHT_DECAY']),
                                               name=params['CLASSIFIER_ACTIVATION'],
                                               trainable=self.trainable),
                                         name='target-text')

        softoutQE = shared_QE_soft(out_layer)
        softout = shared_FC_soft(softoutQE)

        trg_words = Input(name=self.ids_inputs[3],
                          batch_shape=tuple([None, params['SECOND_DIM_SIZE'], params['MAX_INPUT_TEXT_LEN']]), dtype='int32')
        trg_words_in = genreshape(trg_words)

        next_words_one_hot = one_hot(trg_words_in, params)

        # we multiply the weight matrix by one-hot reference vector to make vectors for the words we don't need zeroes
        qv_prep = DenseTranspose(params['QE_VECTOR_SIZE'], shared_FC_soft, self.ids_outputs[0], name='transpose',
                                 trainable=self.trainable_est)

        qv_prep_res = qv_prep(next_words_one_hot)

        # we multiply our matrix with zero values by our quality vectors so that zero vectors do not influence our decisions
        qv = multiply([qv_prep_res, softoutQE], name='qv')

        enc_qe_frw, last_state_frw = GRU(params['QE_VECTOR_SIZE'], return_sequences=True, return_state=True,
                                         name='qe_frw', trainable=self.trainable_est)(qv)
        enc_qe_bkw, last_state_bkw = GRU(params['QE_VECTOR_SIZE'], return_sequences=True, return_state=True,
                                         go_backwards=True, name='qe_bkw', trainable=self.trainable_est)(qv)

        enc_qe_bkw = Reverse(enc_qe_bkw._keras_shape[2], axes=1, trainable=self.trainable_est,
                             name='reverse_enc_qe_bkw')(enc_qe_bkw)

        enc_qe_concat = concatenate([enc_qe_frw, enc_qe_bkw], name='enc_qe_concat')
        last_state_concat = concatenate([last_state_frw, last_state_bkw], trainable=self.trainable_est,
                                        name='last_state_concat')

        # reshape back to 3d input to group sent vectors per doc
        genreshape_out = GeneralReshape((None, params['SECOND_DIM_SIZE'], params['QE_VECTOR_SIZE'] * 2), params)

        last_state_concat = genreshape_out(last_state_concat)

        #bi-RNN over doc sentences
        dec_doc_frw, dec_doc_last_state_frw = GRU(params['DOC_DECODER_HIDDEN_SIZE'], return_sequences=True,
                                                  return_state=True,
                                                  name='dec_doc_frw')(last_state_concat)
        dec_doc_bkw, dec_doc_last_state_bkw = GRU(params['DOC_DECODER_HIDDEN_SIZE'], return_sequences=True,
                                                  return_state=True,
                                                  go_backwards=True, name='dec_doc_bkw')(last_state_concat)

        dec_doc_bkw = Reverse(dec_doc_bkw._keras_shape[2], axes=1,
                              name='dec_reverse_doc_bkw')(dec_doc_bkw)

        dec_doc_seq_concat = concatenate([dec_doc_frw, dec_doc_bkw], trainable=self.trainable_est,
                                         name='dec_doc_seq_concat')

        dec_doc_last_state_concat = concatenate([dec_doc_last_state_frw, dec_doc_last_state_bkw],
                                                name='dec_doc_last_state_concat')

        dec_doc_seq_concat  = NonMasking()(dec_doc_seq_concat)

        # apply doc-level attention over sentences
        attention_mul = attention_3d_block(dec_doc_seq_concat, params, 'doc')
        out_activation=params.get('OUT_ACTIVATION', 'sigmoid')

        qe_doc = Dense(1, activation=out_activation, name=self.ids_outputs[0])(attention_mul)

        self.model = Model(inputs=[src_text, next_words, next_words_bkw, trg_words],
                           outputs=[qe_doc])




    #======================================================
    # Sentence-level QE -- POSTECH-inspired Estimator model
    #======================================================
    #
    ## Inputs:
    # 1. Sentences in src language (shape: (mini_batch_size, words))
    # 2. One-position left-shifted machine-translated sentences to represent the right context (shape: (mini_batch_size, words))
    # 3. One-position rigth-shifted machine-translated sentences to represent the left context (shape: (mini_batch_size, words))
    # 4. Unshifted machine-translated sentences for evaluation (shape: (mini_batch_size, words))
    #
    ## Output:
    # 1. Sentence quality scores (shape: (mini_batch_size,))
    #
    ## References:
    # - Hyun Kim, Hun-Young Jung, Hongseok Kwon, Jong-Hyeok Lee, and Seung-Hoon Na. 2017a. Predictor- estimator: Neural quality estimation based on target word prediction for machine translation. ACM Trans. Asian Low-Resour. Lang. Inf. Process., 17(1):3:1-3:22, September.


    def EstimatorSent(self, params):
        # 1. Source text input
        src_text = Input(name=self.ids_inputs[0], batch_shape=tuple([None, None]), dtype='int32')

        # 2. Encoder
        # 2.1. Source word embedding
        src_embedding = Embedding(params['INPUT_VOCABULARY_SIZE'], params['SOURCE_TEXT_EMBEDDING_SIZE'],
                                  name='source_word_embedding',
                                  embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                  embeddings_initializer=params['INIT_FUNCTION'],
                                  trainable=self.trainable,
                                  mask_zero=True)(src_text)
        src_embedding = Regularize(src_embedding, params, trainable=self.trainable,name='src_embedding')

        # 2.2. BRNN encoder (GRU/LSTM)
        if params['BIDIRECTIONAL_ENCODER']:
            annotations = Bidirectional(eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                         kernel_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         recurrent_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         bias_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                                         recurrent_dropout=params[
                                                                             'RECURRENT_DROPOUT_P'],
                                                                         kernel_initializer=params['INIT_FUNCTION'],
                                                                         recurrent_initializer=params['INNER_INIT'],
                                                                         return_sequences=True, trainable=self.trainable),
                                        name='bidirectional_encoder_' + params['ENCODER_RNN_TYPE'],
                                        merge_mode='concat')(src_embedding)
        else:
            annotations = eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                           kernel_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                           recurrent_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                           bias_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                           dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                           recurrent_dropout=params['RECURRENT_DROPOUT_P'],
                                                           kernel_initializer=params['INIT_FUNCTION'],
                                                           recurrent_initializer=params['INNER_INIT'],
                                                           return_sequences=True,
                                                           name='encoder_' + params['ENCODER_RNN_TYPE'],
                                                           trainable=self.trainable)(src_embedding)
        annotations = Regularize(annotations, params, trainable=self.trainable, name='annotations')

        # 2.3. Potentially deep encoder
        for n_layer in range(1, params['N_LAYERS_ENCODER']):
            if params['BIDIRECTIONAL_DEEP_ENCODER']:
                current_annotations = Bidirectional(eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                                     kernel_regularizer=l2(
                                                                                         params[
                                                                                             'RECURRENT_WEIGHT_DECAY']),
                                                                                     recurrent_regularizer=l2(
                                                                                         params[
                                                                                             'RECURRENT_WEIGHT_DECAY']),
                                                                                     bias_regularizer=l2(
                                                                                         params[
                                                                                             'RECURRENT_WEIGHT_DECAY']),
                                                                                     dropout=params[
                                                                                         'RECURRENT_INPUT_DROPOUT_P'],
                                                                                     recurrent_dropout=params[
                                                                                         'RECURRENT_DROPOUT_P'],
                                                                                     kernel_initializer=params[
                                                                                         'INIT_FUNCTION'],
                                                                                     recurrent_initializer=params[
                                                                                         'INNER_INIT'],
                                                                                     return_sequences=True,
                                                                                     trainable=self.trainable,
                                                                                     ),
                                                    merge_mode='concat',
                                                    name='bidirectional_encoder_' + str(n_layer))(annotations)
            else:
                current_annotations = eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                       kernel_regularizer=l2(
                                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                                       recurrent_regularizer=l2(
                                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                                       bias_regularizer=l2(
                                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                                       dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                                       recurrent_dropout=params['RECURRENT_DROPOUT_P'],
                                                                       kernel_initializer=params['INIT_FUNCTION'],
                                                                       recurrent_initializer=params['INNER_INIT'],
                                                                       return_sequences=True,
                                                                       trainable=self.trainable,
                                                                       name='encoder_' + str(n_layer))(annotations)
            current_annotations = Regularize(current_annotations, params, trainable=self.trainable, name='annotations_' + str(n_layer))
            annotations = Add(trainable=self.trainable)([annotations, current_annotations])

        # 3. Decoder
        # 3.1.1. Previously generated words as inputs for training -> Teacher forcing
        #trg_words = Input(name=self.ids_inputs[1], batch_shape=tuple([None, None]), dtype='int32')
        next_words = Input(name=self.ids_inputs[1], batch_shape=tuple([None, None]), dtype='int32')
        next_words_bkw = Input(name=self.ids_inputs[2], batch_shape=tuple([None, None]), dtype='int32')

        # 3.1.2. Target word embedding
        state_below = Embedding(params['OUTPUT_VOCABULARY_SIZE'], params['TARGET_TEXT_EMBEDDING_SIZE'],
                                name='target_word_embedding_below',
                                embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                embeddings_initializer=params['INIT_FUNCTION'],
                                trainable=self.trainable,
                                mask_zero=True)(next_words)
        state_below = Regularize(state_below, params, trainable=self.trainable, name='state_below')

        state_above = Embedding(params['OUTPUT_VOCABULARY_SIZE'], params['TARGET_TEXT_EMBEDDING_SIZE'],
                                name='target_word_embedding_above',
                                embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                embeddings_initializer=params['INIT_FUNCTION'],
                                trainable=self.trainable,
                                mask_zero=True)(next_words_bkw)
        state_above = Regularize(state_above, params, trainable=self.trainable, name='state_above')

        # 3.2. Decoder's RNN initialization perceptrons with ctx mean
        ctx_mean = MaskedMean(trainable=self.trainable)(annotations)
        annotations = MaskLayer(trainable=self.trainable)(annotations)  # We may want the padded annotations

        if len(params['INIT_LAYERS']) > 0:
            for n_layer_init in range(len(params['INIT_LAYERS']) - 1):
                ctx_mean = Dense(params['DECODER_HIDDEN_SIZE'], name='init_layer_%d' % n_layer_init,
                                 kernel_initializer=params['INIT_FUNCTION'],
                                 kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                 bias_regularizer=l2(params['WEIGHT_DECAY']),
                                 activation=params['INIT_LAYERS'][n_layer_init],
                                 trainable=self.trainable
                                 )(ctx_mean)
                ctx_mean = Regularize(ctx_mean, params, trainable=self.trainable, name='ctx' + str(n_layer_init))

            initial_state = Dense(params['DECODER_HIDDEN_SIZE'], name='initial_state',
                                  kernel_initializer=params['INIT_FUNCTION'],
                                  kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                  bias_regularizer=l2(params['WEIGHT_DECAY']),
                                  activation=params['INIT_LAYERS'][-1],
                                  trainable=self.trainable
                                  )(ctx_mean)
            initial_state = Regularize(initial_state, params, trainable=self.trainable, name='initial_state')
            input_attentional_decoder = [state_below, annotations, initial_state]

            if 'LSTM' in params['DECODER_RNN_TYPE']:
                initial_memory = Dense(params['DECODER_HIDDEN_SIZE'], name='initial_memory',
                                       kernel_initializer=params['INIT_FUNCTION'],
                                       kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                       bias_regularizer=l2(params['WEIGHT_DECAY']),
                                       activation=params['INIT_LAYERS'][-1],
                                       trainable=self.trainable)(ctx_mean)
                initial_memory = Regularize(initial_memory, params, trainable=self.trainable, name='initial_memory')
                input_attentional_decoder.append(initial_memory)
        else:
            # Initialize to zeros vector
            input_attentional_decoder = [state_below, annotations]
            initial_state = ZeroesLayer(params['DECODER_HIDDEN_SIZE'],trainable=self.trainable)(ctx_mean)
            input_attentional_decoder.append(initial_state)
            if 'LSTM' in params['DECODER_RNN_TYPE']:
                input_attentional_decoder.append(initial_state)

        # 3.3. Attentional decoder
        sharedAttRNNCond = eval('Att' + params['DECODER_RNN_TYPE'] + 'Cond')(params['DECODER_HIDDEN_SIZE'],
                                                                             att_units=params.get('ATTENTION_SIZE', 0),
                                                                             kernel_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             recurrent_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             conditional_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             bias_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             attention_context_wa_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             attention_recurrent_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             attention_context_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             bias_ba_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             dropout=params[
                                                                                 'RECURRENT_INPUT_DROPOUT_P'],
                                                                             recurrent_dropout=params[
                                                                                 'RECURRENT_DROPOUT_P'],
                                                                             conditional_dropout=params[
                                                                                 'RECURRENT_INPUT_DROPOUT_P'],
                                                                             attention_dropout=params['DROPOUT_P'],
                                                                             kernel_initializer=params['INIT_FUNCTION'],
                                                                             recurrent_initializer=params['INNER_INIT'],
                                                                             attention_context_initializer=params[
                                                                                 'INIT_ATT'],
                                                                             return_sequences=True,
                                                                             return_extra_variables=True,
                                                                             return_states=True,
                                                                             num_inputs=len(input_attentional_decoder),
                                                                             name='decoder_Att' + params[
                                                                                 'DECODER_RNN_TYPE'] + 'Cond',
                                                                             trainable=self.trainable)

        rnn_output = sharedAttRNNCond(input_attentional_decoder)
        proj_h = rnn_output[0]
        x_att = rnn_output[1]
        alphas = rnn_output[2]
        h_state = rnn_output[3]

        trg_enc_frw = eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                     kernel_regularizer=l2(
                                                                         params['RECURRENT_WEIGHT_DECAY']),
                                                                     recurrent_regularizer=l2(
                                                                         params['RECURRENT_WEIGHT_DECAY']),
                                                                     bias_regularizer=l2(
                                                                         params['RECURRENT_WEIGHT_DECAY']),
                                                                     dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                                     recurrent_dropout=params[
                                                                         'RECURRENT_DROPOUT_P'],
                                                                     kernel_initializer=params['INIT_FUNCTION'],
                                                                     recurrent_initializer=params['INNER_INIT'],
                                                                     return_sequences=True,
                                                                     trainable=self.trainable,
                                                                     name='enc_trg_frw')

        trg_enc_bkw = eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                         kernel_regularizer=l2(
                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                         recurrent_regularizer=l2(
                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                         bias_regularizer=l2(
                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                         dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                         recurrent_dropout=params[
                                                             'RECURRENT_DROPOUT_P'],
                                                         kernel_initializer=params['INIT_FUNCTION'],
                                                         recurrent_initializer=params['INNER_INIT'],
                                                         return_sequences=True,
                                                         trainable=self.trainable,
                                                         go_backwards=True,
                                                         name='enc_trg_bkw')

        trg_state_frw = trg_enc_frw(state_below)
        trg_state_bkw = trg_enc_bkw(state_above)

        trg_state_bkw = Reverse(trg_state_bkw._keras_shape[2], axes=1, trainable=self.trainable, name='reverse_trg_state_bkw')(trg_state_bkw)

        # preparing formula 3b
        merged_emb = concatenate([state_below, state_above], axis=2, trainable=self.trainable, name='merged_emb')
        merged_states = concatenate([trg_state_frw, trg_state_bkw], axis=2, trainable=self.trainable, name='merged_states')

        # we replace state before with the concatenation of state before and after
        proj_h = merged_states

        if 'LSTM' in params['DECODER_RNN_TYPE']:
            h_memory = rnn_output[4]
        shared_Lambda_Permute = PermuteGeneral((1, 0, 2),trainable=self.trainable)

        if params['DOUBLE_STOCHASTIC_ATTENTION_REG'] > 0:
            alpha_regularizer = AlphaRegularizer(alpha_factor=params['DOUBLE_STOCHASTIC_ATTENTION_REG'])(alphas)

        [proj_h, shared_reg_proj_h] = Regularize(proj_h, params, trainable=self.trainable, shared_layers=True, name='proj_h0')

        # 3.4. Possibly deep decoder
        shared_proj_h_list = []
        shared_reg_proj_h_list = []

        h_states_list = [h_state]
        if 'LSTM' in params['DECODER_RNN_TYPE']:
            h_memories_list = [h_memory]

        for n_layer in range(1, params['N_LAYERS_DECODER']):
            current_rnn_input = [merged_states, shared_Lambda_Permute(x_att), initial_state]
            shared_proj_h_list.append(eval(params['DECODER_RNN_TYPE'].replace('Conditional', '') + 'Cond')(
                params['DECODER_HIDDEN_SIZE'],
                kernel_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                recurrent_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                conditional_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                bias_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                dropout=params['RECURRENT_DROPOUT_P'],
                recurrent_dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                conditional_dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                kernel_initializer=params['INIT_FUNCTION'],
                recurrent_initializer=params['INNER_INIT'],
                return_sequences=True,
                return_states=True,
                num_inputs=len(current_rnn_input),
                trainable=self.trainable,
                name='decoder_' + params['DECODER_RNN_TYPE'].replace(
                    'Conditional', '') + 'Cond' + str(n_layer)))

            if 'LSTM' in params['DECODER_RNN_TYPE']:
                current_rnn_input.append(initial_memory)
            current_rnn_output = shared_proj_h_list[-1](current_rnn_input)
            current_proj_h = current_rnn_output[0]
            h_states_list.append(current_rnn_output[1])
            if 'LSTM' in params['DECODER_RNN_TYPE']:
                h_memories_list.append(current_rnn_output[2])
            [current_proj_h, shared_reg_proj_h] = Regularize(current_proj_h, params, trainable=self.trainable, shared_layers=True,
                                                             name='proj_h' + str(n_layer))
            shared_reg_proj_h_list.append(shared_reg_proj_h)

            proj_h = Add(trainable=self.trainable)([proj_h, current_proj_h])

        # 3.5. Skip connections between encoder and output layer
        shared_FC_mlp = TimeDistributed(Dense(params['SKIP_VECTORS_HIDDEN_SIZE'],
                                              kernel_initializer=params['INIT_FUNCTION'],
                                              kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                              bias_regularizer=l2(params['WEIGHT_DECAY']),
                                              activation='linear',
                                              trainable=self.trainable),
                                        name='logit_lstm')
        out_layer_mlp = shared_FC_mlp(proj_h)
        shared_FC_ctx = TimeDistributed(Dense(params['SKIP_VECTORS_HIDDEN_SIZE'],
                                              kernel_initializer=params['INIT_FUNCTION'],
                                              kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                              bias_regularizer=l2(params['WEIGHT_DECAY']),
                                              activation='linear',
                                              trainable=self.trainable),
                                        name='logit_ctx')
        out_layer_ctx = shared_FC_ctx(x_att)
        out_layer_ctx = shared_Lambda_Permute(out_layer_ctx)
        shared_FC_emb = TimeDistributed(Dense(params['SKIP_VECTORS_HIDDEN_SIZE'],
                                              kernel_initializer=params['INIT_FUNCTION'],
                                              kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                              bias_regularizer=l2(params['WEIGHT_DECAY']),
                                              activation='linear',
                                              trainable=self.trainable),
                                        name='logit_emb')
        out_layer_emb = shared_FC_emb(merged_emb)

        [out_layer_mlp, shared_reg_out_layer_mlp] = Regularize(out_layer_mlp, params,
                                                               shared_layers=True, trainable=self.trainable, name='out_layer_mlp')
        [out_layer_ctx, shared_reg_out_layer_ctx] = Regularize(out_layer_ctx, params,
                                                               shared_layers=True, trainable=self.trainable, name='out_layer_ctx')
        [out_layer_emb, shared_reg_out_layer_emb] = Regularize(out_layer_emb, params,
                                                               shared_layers=True, trainable=self.trainable, name='out_layer_emb')

        shared_additional_output_merge = eval(params['ADDITIONAL_OUTPUT_MERGE_MODE'])(name='additional_input',
                                                                                      trainable=self.trainable)

        # formula 3b addition
        additional_output = shared_additional_output_merge([out_layer_mlp, out_layer_ctx, out_layer_emb])
        shared_activation_tanh = Activation('tanh', trainable=self.trainable)

        out_layer = shared_activation_tanh(additional_output)

        shared_deep_list = []
        shared_reg_deep_list = []

        # 3.6 Optional deep ouput layer
        for i, (activation, dimension) in enumerate(params['DEEP_OUTPUT_LAYERS']):
            shared_deep_list.append(TimeDistributed(Dense(dimension, activation=activation,
                                                          kernel_initializer=params['INIT_FUNCTION'],
                                                          kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                                          bias_regularizer=l2(params['WEIGHT_DECAY']),
                                                          trainable=self.trainable),
                                                    name=activation + '_%d' % i))
            out_layer = shared_deep_list[-1](out_layer)
            [out_layer, shared_reg_out_layer] = Regularize(out_layer,
                                                           params, trainable=self.trainable, shared_layers=True,
                                                           name='out_layer_' + str(activation) + '_%d' % i)
            shared_reg_deep_list.append(shared_reg_out_layer)

        shared_QE_soft = TimeDistributed(Dense(params['QE_VECTOR_SIZE'],
                                               use_bias=False,
                                               kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                               bias_regularizer=l2(params['WEIGHT_DECAY']),
                                               name='QE'+params['CLASSIFIER_ACTIVATION'],
                                               trainable=self.trainable),
                                         name='QE'+'target_text')

        # 3.7. Output layer: Softmax
        shared_FC_soft = TimeDistributed(Dense(params['OUTPUT_VOCABULARY_SIZE'],
                                               use_bias=False,
                                               activation=params['CLASSIFIER_ACTIVATION'],
                                               kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                               bias_regularizer=l2(params['WEIGHT_DECAY']),
                                               name=params['CLASSIFIER_ACTIVATION'],
                                               trainable = self.trainable),
                                         name='target-text')

        softoutQE = shared_QE_soft(out_layer)
        softout = shared_FC_soft(softoutQE)

        trg_words = Input(name=self.ids_inputs[3], batch_shape=tuple([None, None]), dtype='int32')

        next_words_one_hot = one_hot(trg_words, params)

        # we multiply the weight matrix by one-hot reference vector to make vectors for the words we don't need zeroes
        qv_prep = DenseTranspose(params['QE_VECTOR_SIZE'], shared_FC_soft, self.ids_outputs[0], name='transpose',
                                 trainable=self.trainable_est)

        qv_prep_res = qv_prep(next_words_one_hot)

        # we multiply our matrix with zero values by our quality vectors so that zero vectors do not influence our decisions
        qv = multiply([qv_prep_res, softoutQE], name='qv')

        enc_qe_frw, last_state_frw = GRU(params['QE_VECTOR_SIZE'], return_sequences=True, return_state=True,
                                         name='qe_frw', trainable=self.trainable_est)(qv)
        enc_qe_bkw, last_state_bkw = GRU(params['QE_VECTOR_SIZE'], return_sequences=True, return_state=True,
                                         go_backwards=True, name='qe_bkw', trainable=self.trainable_est)(qv)

        enc_qe_bkw = Reverse(enc_qe_bkw._keras_shape[2], axes=1, trainable=self.trainable_est,
                             name='reverse_enc_qe_bkw')(enc_qe_bkw)

        last_state_concat = concatenate([last_state_frw, last_state_bkw], trainable=self.trainable_est, name='last_state_concat')

        seq_concat = concatenate([enc_qe_frw, enc_qe_bkw], trainable=self.trainable_est, name='seq_concat')

        # uncomment for Post QE
        # fin_seq = concatenate([seq_concat, merged_states])
        out_activation=params.get('OUT_ACTIVATION', 'sigmoid')
        qe_sent = Dense(1, activation=out_activation, trainable=self.trainable_est, name=self.ids_outputs[0])(last_state_concat)
        #word_qe = TimeDistributed(Dense(params['WORD_QE_CLASSES'], activation='sigmoid'), name=self.ids_outputs[2])(
        #    seq_concat)

        # self.model = Model(inputs=[src_text, next_words, next_words_bkw], outputs=[merged_states,softout, softoutQE])
        # if params['DOUBLE_STOCHASTIC_ATTENTION_REG'] > 0.:
        #     self.model.add_loss(alpha_regularizer)
        self.model = Model(inputs=[src_text, next_words, next_words_bkw, trg_words],
                           outputs=[qe_sent])


    #=============================
    # Phrase-level QE -- BiRNN model
    #=============================
    #
    ## Inputs:
    # 1. Sentences in src language (shape: (mini_batch_size, line_words))
    # 2. Parallel machine-translated sentences (shape: (mini_batch_size, sentence_phrases, words))
    #
    ## Output:
    # 1. Phrase quality labels (shape: (mini_batch_size, number_of_qe_labels))
    #
    ## Summary of the model:
    # The encoder encodes the source, the decoder at each timestep produces an output representation taking into account the previously produced representations, as well as the sum of source word representations weighted by the attention mechanism.
    # The resulting word-level representations are summarized (sum or average) into phrase-level representations used to make classification decisions.


    def EncPhraseAtt(self, params):
        # 1. Source text input
        src_text = Input(name=self.ids_inputs[0], batch_shape=tuple([None, None]), dtype='int32')

        # 2. Encoder
        # 2.1. Source word embedding
        src_embedding = Embedding(params['INPUT_VOCABULARY_SIZE'], params['SOURCE_TEXT_EMBEDDING_SIZE'],
                                  name='source_word_embedding',
                                  embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                  embeddings_initializer=params['INIT_FUNCTION'],
                                  trainable=self.trainable,
                                  mask_zero=True)(src_text)
        src_embedding = Regularize(src_embedding, params, trainable=self.trainable, name='src_embedding')

        # 2.2. BRNN encoder (GRU/LSTM)
        if params['BIDIRECTIONAL_ENCODER']:
            annotations = Bidirectional(eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                         kernel_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         recurrent_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         bias_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                                         recurrent_dropout=params[
                                                                             'RECURRENT_DROPOUT_P'],
                                                                         kernel_initializer=params['INIT_FUNCTION'],
                                                                         recurrent_initializer=params['INNER_INIT'],
                                                                         return_sequences=True,
                                                                         trainable=self.trainable),
                                        name='bidirectional_encoder_' + params['ENCODER_RNN_TYPE'],
                                        merge_mode='concat')(src_embedding)
        else:
            annotations = eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                           kernel_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                           recurrent_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                           bias_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                           dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                           recurrent_dropout=params['RECURRENT_DROPOUT_P'],
                                                           kernel_initializer=params['INIT_FUNCTION'],
                                                           recurrent_initializer=params['INNER_INIT'],
                                                           return_sequences=True,
                                                           name='encoder_' + params['ENCODER_RNN_TYPE'],
                                                           trainable=self.trainable)(src_embedding)
        annotations = Regularize(annotations, params, trainable=self.trainable, name='annotations')

        # 2.3. Potentially deep encoder
        for n_layer in range(1, params['N_LAYERS_ENCODER']):
            if params['BIDIRECTIONAL_DEEP_ENCODER']:
                current_annotations = Bidirectional(eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                                     kernel_regularizer=l2(
                                                                                         params[
                                                                                             'RECURRENT_WEIGHT_DECAY']),
                                                                                     recurrent_regularizer=l2(
                                                                                         params[
                                                                                             'RECURRENT_WEIGHT_DECAY']),
                                                                                     bias_regularizer=l2(
                                                                                         params[
                                                                                             'RECURRENT_WEIGHT_DECAY']),
                                                                                     dropout=params[
                                                                                         'RECURRENT_INPUT_DROPOUT_P'],
                                                                                     recurrent_dropout=params[
                                                                                         'RECURRENT_DROPOUT_P'],
                                                                                     kernel_initializer=params[
                                                                                         'INIT_FUNCTION'],
                                                                                     recurrent_initializer=params[
                                                                                         'INNER_INIT'],
                                                                                     return_sequences=True,
                                                                                     trainable=self.trainable,
                                                                                     ),
                                                    merge_mode='concat',
                                                    name='bidirectional_encoder_' + str(n_layer))(annotations)
            else:
                current_annotations = eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                       kernel_regularizer=l2(
                                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                                       recurrent_regularizer=l2(
                                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                                       bias_regularizer=l2(
                                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                                       dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                                       recurrent_dropout=params['RECURRENT_DROPOUT_P'],
                                                                       kernel_initializer=params['INIT_FUNCTION'],
                                                                       recurrent_initializer=params['INNER_INIT'],
                                                                       return_sequences=True,
                                                                       trainable=self.trainable,
                                                                       name='encoder_' + str(n_layer))(annotations)
            current_annotations = Regularize(current_annotations, params, trainable=self.trainable,
                                             name='annotations_' + str(n_layer))
            annotations = Add(trainable=self.trainable)([annotations, current_annotations])

        # 3. Decoder
        # 3.1.1. Previously generated words as inputs for training -> Teacher forcing
        # trg_words = Input(name=self.ids_inputs[1], batch_shape=tuple([None, None]), dtype='int32')
        trg_words = Input(name=self.ids_inputs[1], batch_shape=tuple([None, params['SECOND_DIM_SIZE'], None]), dtype='int32')
        trg_reshape = Reshape((-1,))
        # reshape MT input to 2d
        trg_words_reshaped = trg_reshape(trg_words)

        # 3.1.2. Target word embedding
        state_below = Embedding(params['OUTPUT_VOCABULARY_SIZE'], params['TARGET_TEXT_EMBEDDING_SIZE'],
                                name='target_word_embedding_below',
                                embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                embeddings_initializer=params['INIT_FUNCTION'],
                                trainable=self.trainable,
                                mask_zero=True)(trg_words_reshaped)
        state_below = Regularize(state_below, params, trainable=self.trainable, name='state_below')

        # 3.2. Decoder's RNN initialization perceptrons with ctx mean
        ctx_mean = MaskedMean(trainable=self.trainable)(annotations)
        annotations = MaskLayer(trainable=self.trainable)(annotations)  # We may want the padded annotations

        if len(params['INIT_LAYERS']) > 0:
            for n_layer_init in range(len(params['INIT_LAYERS']) - 1):
                ctx_mean = Dense(params['DECODER_HIDDEN_SIZE'], name='init_layer_%d' % n_layer_init,
                                 kernel_initializer=params['INIT_FUNCTION'],
                                 kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                 bias_regularizer=l2(params['WEIGHT_DECAY']),
                                 activation=params['INIT_LAYERS'][n_layer_init],
                                 trainable=self.trainable
                                 )(ctx_mean)
                ctx_mean = Regularize(ctx_mean, params, trainable=self.trainable, name='ctx' + str(n_layer_init))

            initial_state = Dense(params['DECODER_HIDDEN_SIZE'], name='initial_state',
                                  kernel_initializer=params['INIT_FUNCTION'],
                                  kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                  bias_regularizer=l2(params['WEIGHT_DECAY']),
                                  activation=params['INIT_LAYERS'][-1],
                                  trainable=self.trainable
                                  )(ctx_mean)
            initial_state = Regularize(initial_state, params, trainable=self.trainable, name='initial_state')
            input_attentional_decoder = [state_below, annotations, initial_state]

            if 'LSTM' in params['DECODER_RNN_TYPE']:
                initial_memory = Dense(params['DECODER_HIDDEN_SIZE'], name='initial_memory',
                                       kernel_initializer=params['INIT_FUNCTION'],
                                       kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                       bias_regularizer=l2(params['WEIGHT_DECAY']),
                                       activation=params['INIT_LAYERS'][-1],
                                       trainable=self.trainable)(ctx_mean)
                initial_memory = Regularize(initial_memory, params, trainable=self.trainable, name='initial_memory')
                input_attentional_decoder.append(initial_memory)
        else:
            # Initialize to zeros vector
            input_attentional_decoder = [state_below, annotations]
            initial_state = ZeroesLayer(params['DECODER_HIDDEN_SIZE'], trainable=self.trainable)(ctx_mean)
            input_attentional_decoder.append(initial_state)
            if 'LSTM' in params['DECODER_RNN_TYPE']:
                input_attentional_decoder.append(initial_state)

        # 3.3. Attentional decoder
        sharedAttRNNCond = eval('Att' + params['DECODER_RNN_TYPE'] + 'Cond')(params['DECODER_HIDDEN_SIZE'],
                                                                             att_units=params.get('ATTENTION_SIZE', 0),
                                                                             kernel_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             recurrent_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             conditional_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             bias_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             attention_context_wa_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             attention_recurrent_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             attention_context_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             bias_ba_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             dropout=params[
                                                                                 'RECURRENT_INPUT_DROPOUT_P'],
                                                                             recurrent_dropout=params[
                                                                                 'RECURRENT_DROPOUT_P'],
                                                                             conditional_dropout=params[
                                                                                 'RECURRENT_INPUT_DROPOUT_P'],
                                                                             attention_dropout=params['DROPOUT_P'],
                                                                             kernel_initializer=params['INIT_FUNCTION'],
                                                                             recurrent_initializer=params['INNER_INIT'],
                                                                             attention_context_initializer=params[
                                                                                 'INIT_ATT'],
                                                                             return_sequences=True,
                                                                             return_extra_variables=True,
                                                                             return_states=True,
                                                                             num_inputs=len(input_attentional_decoder),
                                                                             name='decoder_Att' + params[
                                                                                 'DECODER_RNN_TYPE'] + 'Cond',
                                                                             trainable=self.trainable)

        rnn_output = sharedAttRNNCond(input_attentional_decoder)
        proj_h = rnn_output[0]
        x_att = rnn_output[1]
        alphas = rnn_output[2]
        h_state = rnn_output[3]

        trg_annotations = proj_h
        annotations = NonMasking()(trg_annotations)
        # reshape back to 3d
        annotations_reshape = Reshape((params['SECOND_DIM_SIZE'], -1, params['DECODER_HIDDEN_SIZE']))
        annotations_reshaped = annotations_reshape(annotations)
        #summarize phrase representations (average by default)
        merge_mode = params.get('WORD_MERGE_MODE', None)
        merge_words = Lambda(mask_aware_mean4d, mask_aware_merge_output_shape4d)
        if merge_mode == 'sum':
            merge_words = Lambda(sum4d, mask_aware_merge_output_shape4d)
            
        output_annotations = merge_words(annotations_reshaped)
        out_activation=params.get('OUT_ACTIVATION', 'sigmoid')
        phrase_qe = TimeDistributed(Dense(params['PHRASE_QE_CLASSES'], activation=out_activation), trainable=self.trainable,
                                  name=self.ids_outputs[0])(output_annotations)

        self.model = Model(inputs=[src_text, trg_words],
                           outputs=[phrase_qe])

    #======================================================
    # Phrase-level QE -- POSTECH-inspired Estimator model
    #======================================================
    #
    ## Inputs:
    # 1. Sentences in src language (shape: (mini_batch_size, line_words))
    # 2. Parallel machine-translated sentences (shape: (mini_batch_size, sentence_phrases, words))
    # 3. One-position rigth-shifted machine-translated sentences to represent the left context (shape: (mini_batch_size, sentence_phrases, words))
    # 4. Unshifted machine-translated sentences for evaluation (shape: (mini_batch_size, sentence_phrases, words))
    #
    ## Output:
    # 1. Phrase quality labels (shape: (mini_batch_size, number_of_qe_labels))
    #
    ## References:
    # - Hyun Kim, Hun-Young Jung, Hongseok Kwon, Jong-Hyeok Lee, and Seung-Hoon Na. 2017a. Predictor- estimator: Neural quality estimation based on target word prediction for machine translation. ACM Trans. Asian Low-Resour. Lang. Inf. Process., 17(1):3:1-3:22, September.

    def EstimatorPhrase(self, params):

        # 1. Source text input
        src_text = Input(name=self.ids_inputs[0], batch_shape=tuple([None, None]), dtype='int32')

        # 2. Encoder
        # 2.1. Source word embedding
        src_embedding = Embedding(params['INPUT_VOCABULARY_SIZE'], params['SOURCE_TEXT_EMBEDDING_SIZE'],
                                  name='source_word_embedding',
                                  embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                  embeddings_initializer=params['INIT_FUNCTION'],
                                  trainable=self.trainable,
                                  mask_zero=True)(src_text)
        src_embedding = Regularize(src_embedding, params, trainable=self.trainable, name='src_embedding')

        # 2.2. BRNN encoder (GRU/LSTM)
        if params['BIDIRECTIONAL_ENCODER']:
            annotations = Bidirectional(eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                         kernel_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         recurrent_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         bias_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                                         recurrent_dropout=params[
                                                                             'RECURRENT_DROPOUT_P'],
                                                                         kernel_initializer=params['INIT_FUNCTION'],
                                                                         recurrent_initializer=params['INNER_INIT'],
                                                                         return_sequences=True,
                                                                         trainable=self.trainable),
                                        name='bidirectional_encoder_' + params['ENCODER_RNN_TYPE'],
                                        merge_mode='concat')(src_embedding)
        else:
            annotations = eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                           kernel_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                           recurrent_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                           bias_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                           dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                           recurrent_dropout=params['RECURRENT_DROPOUT_P'],
                                                           kernel_initializer=params['INIT_FUNCTION'],
                                                           recurrent_initializer=params['INNER_INIT'],
                                                           return_sequences=True,
                                                           name='encoder_' + params['ENCODER_RNN_TYPE'],
                                                           trainable=self.trainable)(src_embedding)
        annotations = Regularize(annotations, params, trainable=self.trainable, name='annotations')

        # 2.3. Potentially deep encoder
        for n_layer in range(1, params['N_LAYERS_ENCODER']):
            if params['BIDIRECTIONAL_DEEP_ENCODER']:
                current_annotations = Bidirectional(eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                                     kernel_regularizer=l2(
                                                                                         params[
                                                                                             'RECURRENT_WEIGHT_DECAY']),
                                                                                     recurrent_regularizer=l2(
                                                                                         params[
                                                                                             'RECURRENT_WEIGHT_DECAY']),
                                                                                     bias_regularizer=l2(
                                                                                         params[
                                                                                             'RECURRENT_WEIGHT_DECAY']),
                                                                                     dropout=params[
                                                                                         'RECURRENT_INPUT_DROPOUT_P'],
                                                                                     recurrent_dropout=params[
                                                                                         'RECURRENT_DROPOUT_P'],
                                                                                     kernel_initializer=params[
                                                                                         'INIT_FUNCTION'],
                                                                                     recurrent_initializer=params[
                                                                                         'INNER_INIT'],
                                                                                     return_sequences=True,
                                                                                     trainable=self.trainable,
                                                                                     ),
                                                    merge_mode='concat',
                                                    name='bidirectional_encoder_' + str(n_layer))(annotations)
            else:
                current_annotations = eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                       kernel_regularizer=l2(
                                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                                       recurrent_regularizer=l2(
                                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                                       bias_regularizer=l2(
                                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                                       dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                                       recurrent_dropout=params['RECURRENT_DROPOUT_P'],
                                                                       kernel_initializer=params['INIT_FUNCTION'],
                                                                       recurrent_initializer=params['INNER_INIT'],
                                                                       return_sequences=True,
                                                                       trainable=self.trainable,
                                                                       name='encoder_' + str(n_layer))(annotations)
            current_annotations = Regularize(current_annotations, params, trainable=self.trainable,
                                             name='annotations_' + str(n_layer))
            annotations = Add(trainable=self.trainable)([annotations, current_annotations])

        # 3. Decoder
        # 3.1.1. Previously generated words as inputs for training -> Teacher forcing
        #trg_words = Input(name=self.ids_inputs[1], batch_shape=tuple([None, None]), dtype='int32')
        next_words = Input(name=self.ids_inputs[1], batch_shape=tuple([None, params['SECOND_DIM_SIZE'], None]), dtype='int32')

        # Reshape phrase inputs to 2d
        trg_reshape = Reshape((-1, ))
        next_words_reshaped = trg_reshape(next_words)

        next_words_bkw = Input(name=self.ids_inputs[2], batch_shape=tuple([None, params['SECOND_DIM_SIZE'], None]), dtype='int32')
        next_words_bkw_reshaped = trg_reshape(next_words_bkw)
        # 3.1.2. Target word embedding
        state_below = Embedding(params['OUTPUT_VOCABULARY_SIZE'], params['TARGET_TEXT_EMBEDDING_SIZE'],
                                name='target_word_embedding_below',
                                embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                embeddings_initializer=params['INIT_FUNCTION'],
                                trainable=self.trainable,
                                mask_zero=True)(next_words_reshaped)
        state_below = Regularize(state_below, params, trainable=self.trainable, name='state_below')

        state_above = Embedding(params['OUTPUT_VOCABULARY_SIZE'], params['TARGET_TEXT_EMBEDDING_SIZE'],
                                name='target_word_embedding_above',
                                embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                embeddings_initializer=params['INIT_FUNCTION'],
                                trainable=self.trainable,
                                mask_zero=True)(next_words_bkw_reshaped)
        state_above = Regularize(state_above, params, trainable=self.trainable, name='state_above')

        # 3.2. Decoder's RNN initialization perceptrons with ctx mean
        ctx_mean = MaskedMean(trainable=self.trainable)(annotations)
        annotations = MaskLayer(trainable=self.trainable)(annotations)  # We may want the padded annotations

        if len(params['INIT_LAYERS']) > 0:
            for n_layer_init in range(len(params['INIT_LAYERS']) - 1):
                ctx_mean = Dense(params['DECODER_HIDDEN_SIZE'], name='init_layer_%d' % n_layer_init,
                                 kernel_initializer=params['INIT_FUNCTION'],
                                 kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                 bias_regularizer=l2(params['WEIGHT_DECAY']),
                                 activation=params['INIT_LAYERS'][n_layer_init],
                                 trainable=self.trainable
                                 )(ctx_mean)
                ctx_mean = Regularize(ctx_mean, params, trainable=self.trainable, name='ctx' + str(n_layer_init))

            initial_state = Dense(params['DECODER_HIDDEN_SIZE'], name='initial_state',
                                  kernel_initializer=params['INIT_FUNCTION'],
                                  kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                  bias_regularizer=l2(params['WEIGHT_DECAY']),
                                  activation=params['INIT_LAYERS'][-1],
                                  trainable=self.trainable
                                  )(ctx_mean)
            initial_state = Regularize(initial_state, params, trainable=self.trainable, name='initial_state')
            input_attentional_decoder = [state_below, annotations, initial_state]

            if 'LSTM' in params['DECODER_RNN_TYPE']:
                initial_memory = Dense(params['DECODER_HIDDEN_SIZE'], name='initial_memory',
                                       kernel_initializer=params['INIT_FUNCTION'],
                                       kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                       bias_regularizer=l2(params['WEIGHT_DECAY']),
                                       activation=params['INIT_LAYERS'][-1],
                                       trainable=self.trainable)(ctx_mean)
                initial_memory = Regularize(initial_memory, params, trainable=self.trainable, name='initial_memory')
                input_attentional_decoder.append(initial_memory)
        else:
            # Initialize to zeros vector
            input_attentional_decoder = [state_below, annotations]
            initial_state = ZeroesLayer(params['DECODER_HIDDEN_SIZE'], trainable=self.trainable)(ctx_mean)
            input_attentional_decoder.append(initial_state)
            if 'LSTM' in params['DECODER_RNN_TYPE']:
                input_attentional_decoder.append(initial_state)

        # 3.3. Attentional decoder
        sharedAttRNNCond = eval('Att' + params['DECODER_RNN_TYPE'] + 'Cond')(params['DECODER_HIDDEN_SIZE'],
                                                                             att_units=params.get('ATTENTION_SIZE', 0),
                                                                             kernel_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             recurrent_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             conditional_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             bias_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             attention_context_wa_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             attention_recurrent_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             attention_context_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             bias_ba_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             dropout=params[
                                                                                 'RECURRENT_INPUT_DROPOUT_P'],
                                                                             recurrent_dropout=params[
                                                                                 'RECURRENT_DROPOUT_P'],
                                                                             conditional_dropout=params[
                                                                                 'RECURRENT_INPUT_DROPOUT_P'],
                                                                             attention_dropout=params['DROPOUT_P'],
                                                                             kernel_initializer=params['INIT_FUNCTION'],
                                                                             recurrent_initializer=params['INNER_INIT'],
                                                                             attention_context_initializer=params[
                                                                                 'INIT_ATT'],
                                                                             return_sequences=True,
                                                                             return_extra_variables=True,
                                                                             return_states=True,
                                                                             num_inputs=len(input_attentional_decoder),
                                                                             name='decoder_Att' + params[
                                                                                 'DECODER_RNN_TYPE'] + 'Cond',
                                                                             trainable=self.trainable)

        rnn_output = sharedAttRNNCond(input_attentional_decoder)
        proj_h = rnn_output[0]
        x_att = rnn_output[1]
        alphas = rnn_output[2]
        h_state = rnn_output[3]

        trg_enc_frw = eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                       kernel_regularizer=l2(
                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                       recurrent_regularizer=l2(
                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                       bias_regularizer=l2(
                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                       dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                       recurrent_dropout=params[
                                                           'RECURRENT_DROPOUT_P'],
                                                       kernel_initializer=params['INIT_FUNCTION'],
                                                       recurrent_initializer=params['INNER_INIT'],
                                                       return_sequences=True,
                                                       trainable=self.trainable,
                                                       name='enc_trg_frw')

        trg_enc_bkw = eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                       kernel_regularizer=l2(
                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                       recurrent_regularizer=l2(
                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                       bias_regularizer=l2(
                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                       dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                       recurrent_dropout=params[
                                                           'RECURRENT_DROPOUT_P'],
                                                       kernel_initializer=params['INIT_FUNCTION'],
                                                       recurrent_initializer=params['INNER_INIT'],
                                                       return_sequences=True,
                                                       trainable=self.trainable,
                                                       go_backwards=True,
                                                       name='enc_trg_bkw')

        trg_state_frw = trg_enc_frw(state_below)
        trg_state_bkw = trg_enc_bkw(state_above)

        trg_state_bkw = Reverse(trg_state_bkw._keras_shape[2], axes=1, trainable=self.trainable,
                                name='reverse_trg_state_bkw')(trg_state_bkw)

        # preparing formula 3b
        merged_emb = concatenate([state_below, state_above], axis=2, trainable=self.trainable, name='merged_emb')
        merged_states = concatenate([trg_state_frw, trg_state_bkw], axis=2, trainable=self.trainable,
                                    name='merged_states')

        # we replace state before with the concatenation of state before and after
        proj_h = merged_states

        if 'LSTM' in params['DECODER_RNN_TYPE']:
            h_memory = rnn_output[4]
        shared_Lambda_Permute = PermuteGeneral((1, 0, 2), trainable=self.trainable)

        if params['DOUBLE_STOCHASTIC_ATTENTION_REG'] > 0:
            alpha_regularizer = AlphaRegularizer(alpha_factor=params['DOUBLE_STOCHASTIC_ATTENTION_REG'])(alphas)

        [proj_h, shared_reg_proj_h] = Regularize(proj_h, params, trainable=self.trainable, shared_layers=True,
                                                 name='proj_h0')

        # 3.4. Possibly deep decoder
        shared_proj_h_list = []
        shared_reg_proj_h_list = []

        h_states_list = [h_state]
        if 'LSTM' in params['DECODER_RNN_TYPE']:
            h_memories_list = [h_memory]

        for n_layer in range(1, params['N_LAYERS_DECODER']):
            current_rnn_input = [merged_states, shared_Lambda_Permute(x_att), initial_state]
            shared_proj_h_list.append(eval(params['DECODER_RNN_TYPE'].replace('Conditional', '') + 'Cond')(
                params['DECODER_HIDDEN_SIZE'],
                kernel_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                recurrent_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                conditional_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                bias_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                dropout=params['RECURRENT_DROPOUT_P'],
                recurrent_dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                conditional_dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                kernel_initializer=params['INIT_FUNCTION'],
                recurrent_initializer=params['INNER_INIT'],
                return_sequences=True,
                return_states=True,
                num_inputs=len(current_rnn_input),
                trainable=self.trainable,
                name='decoder_' + params['DECODER_RNN_TYPE'].replace(
                    'Conditional', '') + 'Cond' + str(n_layer)))

            if 'LSTM' in params['DECODER_RNN_TYPE']:
                current_rnn_input.append(initial_memory)
            current_rnn_output = shared_proj_h_list[-1](current_rnn_input)
            current_proj_h = current_rnn_output[0]
            h_states_list.append(current_rnn_output[1])
            if 'LSTM' in params['DECODER_RNN_TYPE']:
                h_memories_list.append(current_rnn_output[2])
            [current_proj_h, shared_reg_proj_h] = Regularize(current_proj_h, params, trainable=self.trainable,
                                                             shared_layers=True,
                                                             name='proj_h' + str(n_layer))
            shared_reg_proj_h_list.append(shared_reg_proj_h)

            proj_h = Add(trainable=self.trainable)([proj_h, current_proj_h])

        # 3.5. Skip connections between encoder and output layer
        shared_FC_mlp = TimeDistributed(Dense(params['SKIP_VECTORS_HIDDEN_SIZE'],
                                              kernel_initializer=params['INIT_FUNCTION'],
                                              kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                              bias_regularizer=l2(params['WEIGHT_DECAY']),
                                              activation='linear',
                                              trainable=self.trainable),
                                        name='logit_lstm')
        out_layer_mlp = shared_FC_mlp(proj_h)
        shared_FC_ctx = TimeDistributed(Dense(params['SKIP_VECTORS_HIDDEN_SIZE'],
                                              kernel_initializer=params['INIT_FUNCTION'],
                                              kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                              bias_regularizer=l2(params['WEIGHT_DECAY']),
                                              activation='linear',
                                              trainable=self.trainable),
                                        name='logit_ctx')
        out_layer_ctx = shared_FC_ctx(x_att)
        out_layer_ctx = shared_Lambda_Permute(out_layer_ctx)
        shared_FC_emb = TimeDistributed(Dense(params['SKIP_VECTORS_HIDDEN_SIZE'],
                                              kernel_initializer=params['INIT_FUNCTION'],
                                              kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                              bias_regularizer=l2(params['WEIGHT_DECAY']),
                                              activation='linear',
                                              trainable=self.trainable),
                                        name='logit_emb')
        out_layer_emb = shared_FC_emb(merged_emb)

        [out_layer_mlp, shared_reg_out_layer_mlp] = Regularize(out_layer_mlp, params,
                                                               shared_layers=True, trainable=self.trainable,
                                                               name='out_layer_mlp')
        [out_layer_ctx, shared_reg_out_layer_ctx] = Regularize(out_layer_ctx, params,
                                                               shared_layers=True, trainable=self.trainable,
                                                               name='out_layer_ctx')
        [out_layer_emb, shared_reg_out_layer_emb] = Regularize(out_layer_emb, params,
                                                               shared_layers=True, trainable=self.trainable,
                                                               name='out_layer_emb')

        shared_additional_output_merge = eval(params['ADDITIONAL_OUTPUT_MERGE_MODE'])(name='additional_input',
                                                                                      trainable=self.trainable)
        # formula 3b addition
        additional_output = shared_additional_output_merge([out_layer_mlp, out_layer_ctx, out_layer_emb])
        shared_activation_tanh = Activation('tanh', trainable=self.trainable)

        out_layer = shared_activation_tanh(additional_output)

        shared_deep_list = []
        shared_reg_deep_list = []
        # 3.6 Optional deep ouput layer
        for i, (activation, dimension) in enumerate(params['DEEP_OUTPUT_LAYERS']):
            shared_deep_list.append(TimeDistributed(Dense(dimension, activation=activation,
                                                          kernel_initializer=params['INIT_FUNCTION'],
                                                          kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                                          bias_regularizer=l2(params['WEIGHT_DECAY']),
                                                          trainable=self.trainable),
                                                    name=activation + '_%d' % i))
            out_layer = shared_deep_list[-1](out_layer)
            [out_layer, shared_reg_out_layer] = Regularize(out_layer,
                                                           params, trainable=self.trainable, shared_layers=True,
                                                           name='out_layer_' + str(activation) + '_%d' % i)
            shared_reg_deep_list.append(shared_reg_out_layer)

        shared_QE_soft = TimeDistributed(Dense(params['QE_VECTOR_SIZE'],
                                               use_bias=False,
                                               kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                               bias_regularizer=l2(params['WEIGHT_DECAY']),
                                               name='QE' + params['CLASSIFIER_ACTIVATION'],
                                               trainable=self.trainable),
                                         name='QE' + 'target_text')

        # 3.7. Output layer: Softmax
        shared_FC_soft = TimeDistributed(Dense(params['OUTPUT_VOCABULARY_SIZE'],
                                               use_bias=False,
                                               activation=params['CLASSIFIER_ACTIVATION'],
                                               kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                               bias_regularizer=l2(params['WEIGHT_DECAY']),
                                               name=params['CLASSIFIER_ACTIVATION'],
                                               trainable=self.trainable),
                                         name='target-text')

        softoutQE = shared_QE_soft(out_layer)
        softout = shared_FC_soft(softoutQE)

        trg_words = Input(name=self.ids_inputs[3], batch_shape=tuple([None, params['SECOND_DIM_SIZE'], params['MAX_TRG_INPUT_TEXT_LEN']]), dtype='int32')

        #reshape to 2d
        trg_words_reshaped = trg_reshape(trg_words)
        next_words_one_hot = one_hot(trg_words_reshaped, params)

        # we multiply the weight matrix by one-hot reference vector to make vectors for the words we don't need zeroes
        qv_prep = DenseTranspose(params['QE_VECTOR_SIZE'], shared_FC_soft, self.ids_outputs[0], name='transpose',
                                 trainable=self.trainable_est)

        qv_prep_res = qv_prep(next_words_one_hot)

        # we multiply our matrix with zero values by our quality vectors so that zero vectors do not influence our decisions
        qv = multiply([qv_prep_res, softoutQE], name='qv')

        enc_qe_frw, last_state_frw = GRU(params['QE_VECTOR_SIZE'], trainable=self.trainable_est, return_sequences=True, return_state=True,
                                         name='qe_frw')(qv)
        enc_qe_bkw, last_state_bkw = GRU(params['QE_VECTOR_SIZE'], trainable=self.trainable_est, return_sequences=True, return_state=True,
                                         go_backwards=True, name='qe_bkw')(qv)

        enc_qe_bkw = Reverse(enc_qe_bkw._keras_shape[2], axes=1, trainable=self.trainable_est,
                             name='reverse_enc_qe_bkw')(enc_qe_bkw)

        last_state_concat = concatenate([last_state_frw, last_state_bkw], trainable=self.trainable_est, name='last_state_concat')

        seq_concat = concatenate([enc_qe_frw, enc_qe_bkw], trainable=self.trainable_est, name='seq_concat')

        trg_annotations = seq_concat
        annotations = NonMasking()(trg_annotations)

        # reshape back to 3d
        annotations_reshape = Reshape((params['SECOND_DIM_SIZE'], -1, params['QE_VECTOR_SIZE']*2))
        annotations_reshaped = annotations_reshape(annotations)

        #summarize phrase representations (average by default)
        merge_mode = params.get('WORD_MERGE_MODE', None)
        merge_words = Lambda(mask_aware_mean4d, mask_aware_merge_output_shape4d)
        if merge_mode == 'sum':
            merge_words = Lambda(sum4d, mask_aware_merge_output_shape4d)

        output_annotations = merge_words(annotations_reshaped)
        out_activation=params.get('OUT_ACTIVATION', 'sigmoid')
        phrase_qe = TimeDistributed(Dense(params['PHRASE_QE_CLASSES'], activation=out_activation), trainable=self.trainable_est, name=self.ids_outputs[0])(output_annotations)

        self.model = Model(inputs=[src_text, next_words, next_words_bkw, trg_words],
                           outputs=[phrase_qe])
 

    #==================================================
    # Word-level QE -- POSTECH-inspired Estimator model
    #==================================================
    #
    ## Inputs:
    # 1. Sentences in src language (shape: (mini_batch_size, words))
    # 2. One-position left-shifted machine-translated sentences to represent the right context (shape: (mini_batch_size, words))
    # 3. One-position rigth-shifted machine-translated sentences to represent the left context (shape: (mini_batch_size, words))
    # 4. Unshifted machine-translated sentences for evaluation (shape: (mini_batch_size, words))
    #
    ## Output:
    # 1. Word quality labels (shape: (mini_batch_size, number_of_qe_labels))
    #
    ## References:
    # - Hyun Kim, Hun-Young Jung, Hongseok Kwon, Jong-Hyeok Lee, and Seung-Hoon Na. 2017a. Predictor- estimator: Neural quality estimation based on target word prediction for machine translation. ACM Trans. Asian Low-Resour. Lang. Inf. Process., 17(1):3:1-3:22, September.

    def EstimatorWord(self, params):
        # 1. Source text input
        src_text = Input(name=self.ids_inputs[0], batch_shape=tuple([None, None]), dtype='int32')

        # 2. Encoder
        # 2.1. Source word embedding
        src_embedding = Embedding(params['INPUT_VOCABULARY_SIZE'], params['SOURCE_TEXT_EMBEDDING_SIZE'],
                                  name='source_word_embedding',
                                  embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                  embeddings_initializer=params['INIT_FUNCTION'],
                                  trainable=self.trainable,
                                  mask_zero=True)(src_text)
        src_embedding = Regularize(src_embedding, params, trainable=self.trainable, name='src_embedding')

        # 2.2. BRNN encoder (GRU/LSTM)
        if params['BIDIRECTIONAL_ENCODER']:
            annotations = Bidirectional(eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                         kernel_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         recurrent_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         bias_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                                         recurrent_dropout=params[
                                                                             'RECURRENT_DROPOUT_P'],
                                                                         kernel_initializer=params['INIT_FUNCTION'],
                                                                         recurrent_initializer=params['INNER_INIT'],
                                                                         return_sequences=True,
                                                                         trainable=self.trainable),
                                        name='bidirectional_encoder_' + params['ENCODER_RNN_TYPE'],
                                        merge_mode='concat')(src_embedding)
        else:
            annotations = eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                           kernel_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                           recurrent_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                           bias_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                           dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                           recurrent_dropout=params['RECURRENT_DROPOUT_P'],
                                                           kernel_initializer=params['INIT_FUNCTION'],
                                                           recurrent_initializer=params['INNER_INIT'],
                                                           return_sequences=True,
                                                           name='encoder_' + params['ENCODER_RNN_TYPE'],
                                                           trainable=self.trainable)(src_embedding)
        annotations = Regularize(annotations, params, trainable=self.trainable, name='annotations')

        # 2.3. Potentially deep encoder
        for n_layer in range(1, params['N_LAYERS_ENCODER']):
            if params['BIDIRECTIONAL_DEEP_ENCODER']:
                current_annotations = Bidirectional(eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                                     kernel_regularizer=l2(
                                                                                         params[
                                                                                             'RECURRENT_WEIGHT_DECAY']),
                                                                                     recurrent_regularizer=l2(
                                                                                         params[
                                                                                             'RECURRENT_WEIGHT_DECAY']),
                                                                                     bias_regularizer=l2(
                                                                                         params[
                                                                                             'RECURRENT_WEIGHT_DECAY']),
                                                                                     dropout=params[
                                                                                         'RECURRENT_INPUT_DROPOUT_P'],
                                                                                     recurrent_dropout=params[
                                                                                         'RECURRENT_DROPOUT_P'],
                                                                                     kernel_initializer=params[
                                                                                         'INIT_FUNCTION'],
                                                                                     recurrent_initializer=params[
                                                                                         'INNER_INIT'],
                                                                                     return_sequences=True,
                                                                                     trainable=self.trainable,
                                                                                     ),
                                                    merge_mode='concat',
                                                    name='bidirectional_encoder_' + str(n_layer))(annotations)
            else:
                current_annotations = eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                       kernel_regularizer=l2(
                                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                                       recurrent_regularizer=l2(
                                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                                       bias_regularizer=l2(
                                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                                       dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                                       recurrent_dropout=params['RECURRENT_DROPOUT_P'],
                                                                       kernel_initializer=params['INIT_FUNCTION'],
                                                                       recurrent_initializer=params['INNER_INIT'],
                                                                       return_sequences=True,
                                                                       trainable=self.trainable,
                                                                       name='encoder_' + str(n_layer))(annotations)
            current_annotations = Regularize(current_annotations, params, trainable=self.trainable,
                                             name='annotations_' + str(n_layer))
            annotations = Add(trainable=self.trainable)([annotations, current_annotations])

        # 3. Decoder
        # 3.1.1. Previously generated words as inputs for training -> Teacher forcing
        #trg_words = Input(name=self.ids_inputs[1], batch_shape=tuple([None, None]), dtype='int32')
        next_words = Input(name=self.ids_inputs[1], batch_shape=tuple([None, None]), dtype='int32')

        next_words_bkw = Input(name=self.ids_inputs[2], batch_shape=tuple([None, None]), dtype='int32')
        # 3.1.2. Target word embedding
        state_below = Embedding(params['OUTPUT_VOCABULARY_SIZE'], params['TARGET_TEXT_EMBEDDING_SIZE'],
                                name='target_word_embedding_below',
                                embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                embeddings_initializer=params['INIT_FUNCTION'],
                                trainable=self.trainable,
                                mask_zero=True)(next_words)
        state_below = Regularize(state_below, params, trainable=self.trainable, name='state_below')

        state_above = Embedding(params['OUTPUT_VOCABULARY_SIZE'], params['TARGET_TEXT_EMBEDDING_SIZE'],
                                name='target_word_embedding_above',
                                embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                embeddings_initializer=params['INIT_FUNCTION'],
                                trainable=self.trainable,
                                mask_zero=True)(next_words_bkw)
        state_above = Regularize(state_above, params, trainable=self.trainable, name='state_above')

        # 3.2. Decoder's RNN initialization perceptrons with ctx mean
        ctx_mean = MaskedMean(trainable=self.trainable)(annotations)
        annotations = MaskLayer(trainable=self.trainable)(annotations)  # We may want the padded annotations

        if len(params['INIT_LAYERS']) > 0:
            for n_layer_init in range(len(params['INIT_LAYERS']) - 1):
                ctx_mean = Dense(params['DECODER_HIDDEN_SIZE'], name='init_layer_%d' % n_layer_init,
                                 kernel_initializer=params['INIT_FUNCTION'],
                                 kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                 bias_regularizer=l2(params['WEIGHT_DECAY']),
                                 activation=params['INIT_LAYERS'][n_layer_init],
                                 trainable=self.trainable
                                 )(ctx_mean)
                ctx_mean = Regularize(ctx_mean, params, trainable=self.trainable, name='ctx' + str(n_layer_init))

            initial_state = Dense(params['DECODER_HIDDEN_SIZE'], name='initial_state',
                                  kernel_initializer=params['INIT_FUNCTION'],
                                  kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                  bias_regularizer=l2(params['WEIGHT_DECAY']),
                                  activation=params['INIT_LAYERS'][-1],
                                  trainable=self.trainable
                                  )(ctx_mean)
            initial_state = Regularize(initial_state, params, trainable=self.trainable, name='initial_state')
            input_attentional_decoder = [state_below, annotations, initial_state]

            if 'LSTM' in params['DECODER_RNN_TYPE']:
                initial_memory = Dense(params['DECODER_HIDDEN_SIZE'], name='initial_memory',
                                       kernel_initializer=params['INIT_FUNCTION'],
                                       kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                       bias_regularizer=l2(params['WEIGHT_DECAY']),
                                       activation=params['INIT_LAYERS'][-1],
                                       trainable=self.trainable)(ctx_mean)
                initial_memory = Regularize(initial_memory, params, trainable=self.trainable, name='initial_memory')
                input_attentional_decoder.append(initial_memory)
        else:
            # Initialize to zeros vector
            input_attentional_decoder = [state_below, annotations]
            initial_state = ZeroesLayer(params['DECODER_HIDDEN_SIZE'], trainable=self.trainable)(ctx_mean)
            input_attentional_decoder.append(initial_state)
            if 'LSTM' in params['DECODER_RNN_TYPE']:
                input_attentional_decoder.append(initial_state)

        # 3.3. Attentional decoder
        sharedAttRNNCond = eval('Att' + params['DECODER_RNN_TYPE'] + 'Cond')(params['DECODER_HIDDEN_SIZE'],
                                                                             att_units=params.get('ATTENTION_SIZE', 0),
                                                                             kernel_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             recurrent_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             conditional_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             bias_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             attention_context_wa_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             attention_recurrent_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             attention_context_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             bias_ba_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             dropout=params[
                                                                                 'RECURRENT_INPUT_DROPOUT_P'],
                                                                             recurrent_dropout=params[
                                                                                 'RECURRENT_DROPOUT_P'],
                                                                             conditional_dropout=params[
                                                                                 'RECURRENT_INPUT_DROPOUT_P'],
                                                                             attention_dropout=params['DROPOUT_P'],
                                                                             kernel_initializer=params['INIT_FUNCTION'],
                                                                             recurrent_initializer=params['INNER_INIT'],
                                                                             attention_context_initializer=params[
                                                                                 'INIT_ATT'],
                                                                             return_sequences=True,
                                                                             return_extra_variables=True,
                                                                             return_states=True,
                                                                             num_inputs=len(input_attentional_decoder),
                                                                             name='decoder_Att' + params[
                                                                                 'DECODER_RNN_TYPE'] + 'Cond',
                                                                             trainable=self.trainable)

        rnn_output = sharedAttRNNCond(input_attentional_decoder)
        proj_h = rnn_output[0]
        x_att = rnn_output[1]
        alphas = rnn_output[2]
        h_state = rnn_output[3]

        trg_enc_frw = eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                       kernel_regularizer=l2(
                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                       recurrent_regularizer=l2(
                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                       bias_regularizer=l2(
                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                       dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                       recurrent_dropout=params[
                                                           'RECURRENT_DROPOUT_P'],
                                                       kernel_initializer=params['INIT_FUNCTION'],
                                                       recurrent_initializer=params['INNER_INIT'],
                                                       return_sequences=True,
                                                       trainable=self.trainable,
                                                       name='enc_trg_frw')

        trg_enc_bkw = eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                       kernel_regularizer=l2(
                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                       recurrent_regularizer=l2(
                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                       bias_regularizer=l2(
                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                       dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                       recurrent_dropout=params[
                                                           'RECURRENT_DROPOUT_P'],
                                                       kernel_initializer=params['INIT_FUNCTION'],
                                                       recurrent_initializer=params['INNER_INIT'],
                                                       return_sequences=True,
                                                       trainable=self.trainable,
                                                       go_backwards=True,
                                                       name='enc_trg_bkw')

        trg_state_frw = trg_enc_frw(state_below)
        trg_state_bkw = trg_enc_bkw(state_above)

        trg_state_bkw = Reverse(trg_state_bkw._keras_shape[2], axes=1, trainable=self.trainable,
                                name='reverse_trg_state_bkw')(trg_state_bkw)

        # preparing formula 3b
        merged_emb = concatenate([state_below, state_above], axis=2, trainable=self.trainable, name='merged_emb')
        merged_states = concatenate([trg_state_frw, trg_state_bkw], axis=2, trainable=self.trainable,
                                    name='merged_states')

        # we replace state before with the concatenation of state before and after
        proj_h = merged_states

        if 'LSTM' in params['DECODER_RNN_TYPE']:
            h_memory = rnn_output[4]
        shared_Lambda_Permute = PermuteGeneral((1, 0, 2), trainable=self.trainable)

        if params['DOUBLE_STOCHASTIC_ATTENTION_REG'] > 0:
            alpha_regularizer = AlphaRegularizer(alpha_factor=params['DOUBLE_STOCHASTIC_ATTENTION_REG'])(alphas)

        [proj_h, shared_reg_proj_h] = Regularize(proj_h, params, trainable=self.trainable, shared_layers=True,
                                                 name='proj_h0')

        # 3.4. Possibly deep decoder
        shared_proj_h_list = []
        shared_reg_proj_h_list = []

        h_states_list = [h_state]
        if 'LSTM' in params['DECODER_RNN_TYPE']:
            h_memories_list = [h_memory]

        for n_layer in range(1, params['N_LAYERS_DECODER']):
            current_rnn_input = [merged_states, shared_Lambda_Permute(x_att), initial_state]
            shared_proj_h_list.append(eval(params['DECODER_RNN_TYPE'].replace('Conditional', '') + 'Cond')(
                params['DECODER_HIDDEN_SIZE'],
                kernel_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                recurrent_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                conditional_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                bias_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                dropout=params['RECURRENT_DROPOUT_P'],
                recurrent_dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                conditional_dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                kernel_initializer=params['INIT_FUNCTION'],
                recurrent_initializer=params['INNER_INIT'],
                return_sequences=True,
                return_states=True,
                num_inputs=len(current_rnn_input),
                trainable=self.trainable,
                name='decoder_' + params['DECODER_RNN_TYPE'].replace(
                    'Conditional', '') + 'Cond' + str(n_layer)))

            if 'LSTM' in params['DECODER_RNN_TYPE']:
                current_rnn_input.append(initial_memory)
            current_rnn_output = shared_proj_h_list[-1](current_rnn_input)
            current_proj_h = current_rnn_output[0]
            h_states_list.append(current_rnn_output[1])
            if 'LSTM' in params['DECODER_RNN_TYPE']:
                h_memories_list.append(current_rnn_output[2])
            [current_proj_h, shared_reg_proj_h] = Regularize(current_proj_h, params, trainable=self.trainable,
                                                             shared_layers=True,
                                                             name='proj_h' + str(n_layer))
            shared_reg_proj_h_list.append(shared_reg_proj_h)

            proj_h = Add(trainable=self.trainable)([proj_h, current_proj_h])

        # 3.5. Skip connections between encoder and output layer
        shared_FC_mlp = TimeDistributed(Dense(params['SKIP_VECTORS_HIDDEN_SIZE'],
                                              kernel_initializer=params['INIT_FUNCTION'],
                                              kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                              bias_regularizer=l2(params['WEIGHT_DECAY']),
                                              activation='linear',
                                              trainable=self.trainable),
                                        name='logit_lstm')
        out_layer_mlp = shared_FC_mlp(proj_h)
        shared_FC_ctx = TimeDistributed(Dense(params['SKIP_VECTORS_HIDDEN_SIZE'],
                                              kernel_initializer=params['INIT_FUNCTION'],
                                              kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                              bias_regularizer=l2(params['WEIGHT_DECAY']),
                                              activation='linear',
                                              trainable=self.trainable),
                                        name='logit_ctx')
        out_layer_ctx = shared_FC_ctx(x_att)
        out_layer_ctx = shared_Lambda_Permute(out_layer_ctx)
        shared_FC_emb = TimeDistributed(Dense(params['SKIP_VECTORS_HIDDEN_SIZE'],
                                              kernel_initializer=params['INIT_FUNCTION'],
                                              kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                              bias_regularizer=l2(params['WEIGHT_DECAY']),
                                              activation='linear',
                                              trainable=self.trainable),
                                        name='logit_emb')
        out_layer_emb = shared_FC_emb(merged_emb)

        [out_layer_mlp, shared_reg_out_layer_mlp] = Regularize(out_layer_mlp, params,
                                                               shared_layers=True, trainable=self.trainable,
                                                               name='out_layer_mlp')
        [out_layer_ctx, shared_reg_out_layer_ctx] = Regularize(out_layer_ctx, params,
                                                               shared_layers=True, trainable=self.trainable,
                                                               name='out_layer_ctx')
        [out_layer_emb, shared_reg_out_layer_emb] = Regularize(out_layer_emb, params,
                                                               shared_layers=True, trainable=self.trainable,
                                                               name='out_layer_emb')

        shared_additional_output_merge = eval(params['ADDITIONAL_OUTPUT_MERGE_MODE'])(name='additional_input',
                                                                                      trainable=self.trainable)
        # formula 3b addition
        additional_output = shared_additional_output_merge([out_layer_mlp, out_layer_ctx, out_layer_emb])
        shared_activation_tanh = Activation('tanh', trainable=self.trainable)

        out_layer = shared_activation_tanh(additional_output)

        shared_deep_list = []
        shared_reg_deep_list = []
        # 3.6 Optional deep ouput layer
        for i, (activation, dimension) in enumerate(params['DEEP_OUTPUT_LAYERS']):
            shared_deep_list.append(TimeDistributed(Dense(dimension, activation=activation,
                                                          kernel_initializer=params['INIT_FUNCTION'],
                                                          kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                                          bias_regularizer=l2(params['WEIGHT_DECAY']),
                                                          trainable=self.trainable),
                                                    name=activation + '_%d' % i))
            out_layer = shared_deep_list[-1](out_layer)
            [out_layer, shared_reg_out_layer] = Regularize(out_layer,
                                                           params, trainable=self.trainable, shared_layers=True,
                                                           name='out_layer_' + str(activation) + '_%d' % i)
            shared_reg_deep_list.append(shared_reg_out_layer)

        shared_QE_soft = TimeDistributed(Dense(params['QE_VECTOR_SIZE'],
                                               use_bias=False,
                                               kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                               bias_regularizer=l2(params['WEIGHT_DECAY']),
                                               name='QE' + params['CLASSIFIER_ACTIVATION'],
                                               trainable=self.trainable),
                                         name='QE' + 'target_text')

        # 3.7. Output layer: Softmax
        shared_FC_soft = TimeDistributed(Dense(params['OUTPUT_VOCABULARY_SIZE'],
                                               use_bias=False,
                                               activation=params['CLASSIFIER_ACTIVATION'],
                                               kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                               bias_regularizer=l2(params['WEIGHT_DECAY']),
                                               name=params['CLASSIFIER_ACTIVATION'],
                                               trainable=self.trainable),
                                         name='target-text')

        softoutQE = shared_QE_soft(out_layer)
        softout = shared_FC_soft(softoutQE)

        trg_words = Input(name=self.ids_inputs[3], batch_shape=tuple([None, None]), dtype='int32')
        next_words_one_hot = one_hot(trg_words, params)

        # we multiply the weight matrix by one-hot reference vector to make vectors for the words we don't need zeroes
        qv_prep = DenseTranspose(params['QE_VECTOR_SIZE'], shared_FC_soft, self.ids_outputs[0], name='transpose',
                                 trainable=self.trainable_est)

        qv_prep_res = qv_prep(next_words_one_hot)

        # we multiply our matrix with zero values by our quality vectors so that zero vectors do not influence our decisions
        qv = multiply([qv_prep_res, softoutQE], name='qv')

        enc_qe_frw, last_state_frw = GRU(params['QE_VECTOR_SIZE'], trainable=self.trainable_est, return_sequences=True, return_state=True,
                                         name='qe_frw')(qv)
        enc_qe_bkw, last_state_bkw = GRU(params['QE_VECTOR_SIZE'], trainable=self.trainable_est, return_sequences=True, return_state=True,
                                         go_backwards=True, name='qe_bkw')(qv)

        enc_qe_bkw = Reverse(enc_qe_bkw._keras_shape[2], axes=1, trainable=self.trainable_est,
                             name='reverse_enc_qe_bkw')(enc_qe_bkw)

        last_state_concat = concatenate([last_state_frw, last_state_bkw], trainable=self.trainable_est, name='last_state_concat')

        seq_concat = concatenate([enc_qe_frw, enc_qe_bkw], trainable=self.trainable_est, name='seq_concat')

        # uncomment for Post QE
        # fin_seq = concatenate([seq_concat, merged_states])
        #qe_sent = Dense(1, activation='sigmoid', name=self.ids_outputs[0])(last_state_concat)
        out_activation=params.get('OUT_ACTIVATION', 'sigmoid')
        word_qe = TimeDistributed(Dense(params['WORD_QE_CLASSES'], activation=out_activation), trainable=self.trainable_est, name=self.ids_outputs[0])(seq_concat)

        # self.model = Model(inputs=[src_text, next_words, next_words_bkw], outputs=[merged_states,softout, softoutQE])
        # if params['DOUBLE_STOCHASTIC_ATTENTION_REG'] > 0.:
        #     self.model.add_loss(alpha_regularizer)
        self.model = Model(inputs=[src_text, next_words, next_words_bkw, trg_words],
                           outputs=[word_qe])
    


   
    #===============================================================================
    # Word-level QE -- simplified RNN POSTECH model inspired by Jhaveri et al., 2018
    #===============================================================================
    #
    ## Inputs:
    # 1. Sentences in src language (shape: (mini_batch_size, words))
    # 2. Machine-translated sentences (shape: (mini_batch_size, words))
    #
    ## Output:
    # 1. Word quality labels (shape: (mini_batch_size, number_of_qe_labels))
    #
    ## References:
    # - Nisarg Jhaveri, Manish Gupta, and Vasudeva Varman. 2018. Translation quality estimation for indian languages. In Proceedings of th 21st International Conference of the European Association for Machine Transla- tion (EAMT).
    
    def EncWordAtt(self, params):
        # 1. Source text input
        src_text = Input(name=self.ids_inputs[0], batch_shape=tuple([None, None]), dtype='int32')

        # 2. Encoder
        # 2.1. Source word embedding
        src_embedding = Embedding(params['INPUT_VOCABULARY_SIZE'], params['SOURCE_TEXT_EMBEDDING_SIZE'],
                                  name='source_word_embedding',
                                  embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                  embeddings_initializer=params['INIT_FUNCTION'],
                                  trainable=self.trainable,
                                  mask_zero=True)(src_text)
        src_embedding = Regularize(src_embedding, params, trainable=self.trainable, name='src_embedding')

        # 2.2. BRNN encoder (GRU/LSTM)
        if params['BIDIRECTIONAL_ENCODER']:
            annotations = Bidirectional(eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                         kernel_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         recurrent_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         bias_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                                         recurrent_dropout=params[
                                                                             'RECURRENT_DROPOUT_P'],
                                                                         kernel_initializer=params['INIT_FUNCTION'],
                                                                         recurrent_initializer=params['INNER_INIT'],
                                                                         return_sequences=True,
                                                                         trainable=self.trainable),
                                        name='bidirectional_encoder_' + params['ENCODER_RNN_TYPE'],
                                        merge_mode='concat')(src_embedding)
        else:
            annotations = eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                           kernel_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                           recurrent_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                           bias_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                           dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                           recurrent_dropout=params['RECURRENT_DROPOUT_P'],
                                                           kernel_initializer=params['INIT_FUNCTION'],
                                                           recurrent_initializer=params['INNER_INIT'],
                                                           return_sequences=True,
                                                           name='encoder_' + params['ENCODER_RNN_TYPE'],
                                                           trainable=self.trainable)(src_embedding)
        annotations = Regularize(annotations, params, trainable=self.trainable, name='annotations')

        # 2.3. Potentially deep encoder
        for n_layer in range(1, params['N_LAYERS_ENCODER']):
            if params['BIDIRECTIONAL_DEEP_ENCODER']:
                current_annotations = Bidirectional(eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                                     kernel_regularizer=l2(
                                                                                         params[
                                                                                             'RECURRENT_WEIGHT_DECAY']),
                                                                                     recurrent_regularizer=l2(
                                                                                         params[
                                                                                             'RECURRENT_WEIGHT_DECAY']),
                                                                                     bias_regularizer=l2(
                                                                                         params[
                                                                                             'RECURRENT_WEIGHT_DECAY']),
                                                                                     dropout=params[
                                                                                         'RECURRENT_INPUT_DROPOUT_P'],
                                                                                     recurrent_dropout=params[
                                                                                         'RECURRENT_DROPOUT_P'],
                                                                                     kernel_initializer=params[
                                                                                         'INIT_FUNCTION'],
                                                                                     recurrent_initializer=params[
                                                                                         'INNER_INIT'],
                                                                                     return_sequences=True,
                                                                                     trainable=self.trainable,
                                                                                     ),
                                                    merge_mode='concat',
                                                    name='bidirectional_encoder_' + str(n_layer))(annotations)
            else:
                current_annotations = eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                       kernel_regularizer=l2(
                                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                                       recurrent_regularizer=l2(
                                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                                       bias_regularizer=l2(
                                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                                       dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                                       recurrent_dropout=params['RECURRENT_DROPOUT_P'],
                                                                       kernel_initializer=params['INIT_FUNCTION'],
                                                                       recurrent_initializer=params['INNER_INIT'],
                                                                       return_sequences=True,
                                                                       trainable=self.trainable,
                                                                       name='encoder_' + str(n_layer))(annotations)
            current_annotations = Regularize(current_annotations, params, trainable=self.trainable,
                                             name='annotations_' + str(n_layer))
            annotations = Add(trainable=self.trainable)([annotations, current_annotations])

        # 3. Decoder
        # 3.1.1. Previously generated words as inputs for training -> Teacher forcing
        #trg_words = Input(name=self.ids_inputs[1], batch_shape=tuple([None, None]), dtype='int32')
        trg_words = Input(name=self.ids_inputs[1], batch_shape=tuple([None, None]), dtype='int32')

        # 3.1.2. Target word embedding
        state_below = Embedding(params['OUTPUT_VOCABULARY_SIZE'], params['TARGET_TEXT_EMBEDDING_SIZE'],
                                name='target_word_embedding_below',
                                embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                embeddings_initializer=params['INIT_FUNCTION'],
                                trainable=self.trainable,
                                mask_zero=True)(trg_words)
        state_below = Regularize(state_below, params, trainable=self.trainable, name='state_below')

        # 3.2. Decoder's RNN initialization perceptrons with ctx mean
        ctx_mean = MaskedMean(trainable=self.trainable)(annotations)
        annotations = MaskLayer(trainable=self.trainable)(annotations)  # We may want the padded annotations

        if len(params['INIT_LAYERS']) > 0:
            for n_layer_init in range(len(params['INIT_LAYERS']) - 1):
                ctx_mean = Dense(params['DECODER_HIDDEN_SIZE'], name='init_layer_%d' % n_layer_init,
                                 kernel_initializer=params['INIT_FUNCTION'],
                                 kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                 bias_regularizer=l2(params['WEIGHT_DECAY']),
                                 activation=params['INIT_LAYERS'][n_layer_init],
                                 trainable=self.trainable
                                 )(ctx_mean)
                ctx_mean = Regularize(ctx_mean, params, trainable=self.trainable, name='ctx' + str(n_layer_init))

            initial_state = Dense(params['DECODER_HIDDEN_SIZE'], name='initial_state',
                                  kernel_initializer=params['INIT_FUNCTION'],
                                  kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                  bias_regularizer=l2(params['WEIGHT_DECAY']),
                                  activation=params['INIT_LAYERS'][-1],
                                  trainable=self.trainable
                                  )(ctx_mean)
            initial_state = Regularize(initial_state, params, trainable=self.trainable, name='initial_state')
            input_attentional_decoder = [state_below, annotations, initial_state]

            if 'LSTM' in params['DECODER_RNN_TYPE']:
                initial_memory = Dense(params['DECODER_HIDDEN_SIZE'], name='initial_memory',
                                       kernel_initializer=params['INIT_FUNCTION'],
                                       kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                       bias_regularizer=l2(params['WEIGHT_DECAY']),
                                       activation=params['INIT_LAYERS'][-1],
                                       trainable=self.trainable)(ctx_mean)
                initial_memory = Regularize(initial_memory, params, trainable=self.trainable, name='initial_memory')
                input_attentional_decoder.append(initial_memory)
        else:
            # Initialize to zeros vector
            input_attentional_decoder = [state_below, annotations]
            initial_state = ZeroesLayer(params['DECODER_HIDDEN_SIZE'], trainable=self.trainable)(ctx_mean)
            input_attentional_decoder.append(initial_state)
            if 'LSTM' in params['DECODER_RNN_TYPE']:
                input_attentional_decoder.append(initial_state)

        # 3.3. Attentional decoder
        sharedAttRNNCond = eval('Att' + params['DECODER_RNN_TYPE'] + 'Cond')(params['DECODER_HIDDEN_SIZE'],
                                                                             att_units=params.get('ATTENTION_SIZE', 0),
                                                                             kernel_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             recurrent_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             conditional_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             bias_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             attention_context_wa_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             attention_recurrent_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             attention_context_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             bias_ba_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             dropout=params[
                                                                                 'RECURRENT_INPUT_DROPOUT_P'],
                                                                             recurrent_dropout=params[
                                                                                 'RECURRENT_DROPOUT_P'],
                                                                             conditional_dropout=params[
                                                                                 'RECURRENT_INPUT_DROPOUT_P'],
                                                                             attention_dropout=params['DROPOUT_P'],
                                                                             kernel_initializer=params['INIT_FUNCTION'],
                                                                             recurrent_initializer=params['INNER_INIT'],
                                                                             attention_context_initializer=params[
                                                                                 'INIT_ATT'],
                                                                             return_sequences=True,
                                                                             return_extra_variables=True,
                                                                             return_states=True,
                                                                             num_inputs=len(input_attentional_decoder),
                                                                             name='decoder_Att' + params[
                                                                                 'DECODER_RNN_TYPE'] + 'Cond',
                                                                             trainable=self.trainable)

        rnn_output = sharedAttRNNCond(input_attentional_decoder)
        proj_h = rnn_output[0]
        x_att = rnn_output[1]
        alphas = rnn_output[2]
        h_state = rnn_output[3]
        out_activation=params.get('OUT_ACTIVATION', 'sigmoid')

        word_qe = TimeDistributed(Dense(params['WORD_QE_CLASSES'], activation=out_activation), trainable=self.trainable, name=self.ids_outputs[0])(proj_h)

        # self.model = Model(inputs=[src_text, next_words, next_words_bkw], outputs=[merged_states,softout, softoutQE])
        # if params['DOUBLE_STOCHASTIC_ATTENTION_REG'] > 0.:
        #     self.model.add_loss(alpha_regularizer)
        self.model = Model(inputs=[src_text, trg_words],
                           outputs=[word_qe])
 
    #================================
    # POSTECH-inspired Predictor model
    #================================
    #
    ## Inputs:
    # 1. Sentences in src language (shape: (mini_batch_size, words))
    # 2. One-position left-shifted reference sentences to represent the right context (shape: (mini_batch_size, words))
    # 3. One-position rigth-shifted reference sentences to represent the left context (shape: (mini_batch_size, words))
    #
    ## Output:
    # 1. Machine-translated sentences (shape: (mini_batch_size, output_vocabulary_size))
    #
    ## References
    # - Hyun Kim, Hun-Young Jung, Hongseok Kwon, Jong-Hyeok Lee, and Seung-Hoon Na. 2017a. Predictor- estimator: Neural quality estimation based on target word prediction for machine translation. ACM Trans. Asian Low-Resour. Lang. Inf. Process., 17(1):3:1-3:22, September.

    def Predictor(self, params):
        # 1. Source text input
        src_text = Input(name=self.ids_inputs[0], batch_shape=tuple([None, None]), dtype='int32')

        # 2. Encoder
        # 2.1. Source word embedding
        src_embedding = Embedding(params['INPUT_VOCABULARY_SIZE'], params['SOURCE_TEXT_EMBEDDING_SIZE'],
                                  name='source_word_embedding',
                                  embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                  embeddings_initializer=params['INIT_FUNCTION'],
                                  trainable=self.trainable,
                                  mask_zero=True)(src_text)
        src_embedding = Regularize(src_embedding, params, trainable=self.trainable,name='src_embedding')

        # 2.2. BRNN encoder (GRU/LSTM)
        if params['BIDIRECTIONAL_ENCODER']:
            annotations = Bidirectional(eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                         kernel_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         recurrent_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         bias_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                                         recurrent_dropout=params[
                                                                             'RECURRENT_DROPOUT_P'],
                                                                         kernel_initializer=params['INIT_FUNCTION'],
                                                                         recurrent_initializer=params['INNER_INIT'],
                                                                         return_sequences=True, trainable=self.trainable),
                                        name='bidirectional_encoder_' + params['ENCODER_RNN_TYPE'],
                                        merge_mode='concat')(src_embedding)
        else:
            annotations = eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                           kernel_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                           recurrent_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                           bias_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                           dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                           recurrent_dropout=params['RECURRENT_DROPOUT_P'],
                                                           kernel_initializer=params['INIT_FUNCTION'],
                                                           recurrent_initializer=params['INNER_INIT'],
                                                           return_sequences=True,
                                                           name='encoder_' + params['ENCODER_RNN_TYPE'],
                                                           trainable=self.trainable)(src_embedding)
        annotations = Regularize(annotations, params, trainable=self.trainable, name='annotations')
        # 2.3. Potentially deep encoder
        for n_layer in range(1, params['N_LAYERS_ENCODER']):
            if params['BIDIRECTIONAL_DEEP_ENCODER']:
                current_annotations = Bidirectional(eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                                     kernel_regularizer=l2(
                                                                                         params[
                                                                                             'RECURRENT_WEIGHT_DECAY']),
                                                                                     recurrent_regularizer=l2(
                                                                                         params[
                                                                                             'RECURRENT_WEIGHT_DECAY']),
                                                                                     bias_regularizer=l2(
                                                                                         params[
                                                                                             'RECURRENT_WEIGHT_DECAY']),
                                                                                     dropout=params[
                                                                                         'RECURRENT_INPUT_DROPOUT_P'],
                                                                                     recurrent_dropout=params[
                                                                                         'RECURRENT_DROPOUT_P'],
                                                                                     kernel_initializer=params[
                                                                                         'INIT_FUNCTION'],
                                                                                     recurrent_initializer=params[
                                                                                         'INNER_INIT'],
                                                                                     return_sequences=True,
                                                                                     trainable=self.trainable,
                                                                                     ),
                                                    merge_mode='concat',
                                                    name='bidirectional_encoder_' + str(n_layer))(annotations)
            else:
                current_annotations = eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                       kernel_regularizer=l2(
                                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                                       recurrent_regularizer=l2(
                                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                                       bias_regularizer=l2(
                                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                                       dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                                       recurrent_dropout=params['RECURRENT_DROPOUT_P'],
                                                                       kernel_initializer=params['INIT_FUNCTION'],
                                                                       recurrent_initializer=params['INNER_INIT'],
                                                                       return_sequences=True,
                                                                       trainable=self.trainable,
                                                                       name='encoder_' + str(n_layer))(annotations)
            current_annotations = Regularize(current_annotations, params, trainable=self.trainable, name='annotations_' + str(n_layer))
            annotations = Add(trainable=self.trainable)([annotations, current_annotations])

        # 3. Decoder
        # 3.1.1. Previously generated words as inputs for training -> Teacher forcing
        next_words = Input(name=self.ids_inputs[1], batch_shape=tuple([None, None]), dtype='int32')

        next_words_bkw = Input(name=self.ids_inputs[2], batch_shape=tuple([None, None]), dtype='int32')
        # 3.1.2. Target word embedding
        state_below = Embedding(params['OUTPUT_VOCABULARY_SIZE'], params['TARGET_TEXT_EMBEDDING_SIZE'],
                                name='target_word_embedding_below',
                                embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                embeddings_initializer=params['INIT_FUNCTION'],
                                trainable=self.trainable,
                                mask_zero=True)(next_words)
        state_below = Regularize(state_below, params, trainable=self.trainable, name='state_below')

        state_above = Embedding(params['OUTPUT_VOCABULARY_SIZE'], params['TARGET_TEXT_EMBEDDING_SIZE'],
                                name='target_word_embedding_above',
                                embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                embeddings_initializer=params['INIT_FUNCTION'],
                                trainable=self.trainable,
                                mask_zero=True)(next_words_bkw)
        state_above = Regularize(state_above, params, trainable=self.trainable, name='state_above')

        # 3.2. Decoder's RNN initialization perceptrons with ctx mean
        ctx_mean = MaskedMean(trainable=self.trainable)(annotations)
        annotations = MaskLayer(trainable=self.trainable)(annotations)  # We may want the padded annotations

        if len(params['INIT_LAYERS']) > 0:
            for n_layer_init in range(len(params['INIT_LAYERS']) - 1):
                ctx_mean = Dense(params['DECODER_HIDDEN_SIZE'], name='init_layer_%d' % n_layer_init,
                                 kernel_initializer=params['INIT_FUNCTION'],
                                 kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                 bias_regularizer=l2(params['WEIGHT_DECAY']),
                                 activation=params['INIT_LAYERS'][n_layer_init],
                                 trainable=self.trainable
                                 )(ctx_mean)
                ctx_mean = Regularize(ctx_mean, params, trainable=self.trainable, name='ctx' + str(n_layer_init))

            initial_state = Dense(params['DECODER_HIDDEN_SIZE'], name='initial_state',
                                  kernel_initializer=params['INIT_FUNCTION'],
                                  kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                  bias_regularizer=l2(params['WEIGHT_DECAY']),
                                  activation=params['INIT_LAYERS'][-1],
                                  trainable=self.trainable
                                  )(ctx_mean)
            initial_state = Regularize(initial_state, params, trainable=self.trainable, name='initial_state')
            input_attentional_decoder = [state_below, annotations, initial_state]

            if 'LSTM' in params['DECODER_RNN_TYPE']:
                initial_memory = Dense(params['DECODER_HIDDEN_SIZE'], name='initial_memory',
                                       kernel_initializer=params['INIT_FUNCTION'],
                                       kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                       bias_regularizer=l2(params['WEIGHT_DECAY']),
                                       activation=params['INIT_LAYERS'][-1],
                                       trainable=self.trainable)(ctx_mean)
                initial_memory = Regularize(initial_memory, params, trainable=self.trainable, name='initial_memory')
                input_attentional_decoder.append(initial_memory)
        else:
            # Initialize to zeros vector
            input_attentional_decoder = [state_below, annotations]
            initial_state = ZeroesLayer(params['DECODER_HIDDEN_SIZE'],trainable=self.trainable)(ctx_mean)
            input_attentional_decoder.append(initial_state)
            if 'LSTM' in params['DECODER_RNN_TYPE']:
                input_attentional_decoder.append(initial_state)

        # 3.3. Attentional decoder
        sharedAttRNNCond = eval('Att' + params['DECODER_RNN_TYPE'] + 'Cond')(params['DECODER_HIDDEN_SIZE'],
                                                                             att_units=params.get('ATTENTION_SIZE', 0),
                                                                             kernel_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             recurrent_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             conditional_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             bias_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             attention_context_wa_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             attention_recurrent_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             attention_context_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             bias_ba_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             dropout=params[
                                                                                 'RECURRENT_INPUT_DROPOUT_P'],
                                                                             recurrent_dropout=params[
                                                                                 'RECURRENT_DROPOUT_P'],
                                                                             conditional_dropout=params[
                                                                                 'RECURRENT_INPUT_DROPOUT_P'],
                                                                             attention_dropout=params['DROPOUT_P'],
                                                                             kernel_initializer=params['INIT_FUNCTION'],
                                                                             recurrent_initializer=params['INNER_INIT'],
                                                                             attention_context_initializer=params[
                                                                                 'INIT_ATT'],
                                                                             return_sequences=True,
                                                                             return_extra_variables=True,
                                                                             return_states=True,
                                                                             num_inputs=len(input_attentional_decoder),
                                                                             name='decoder_Att' + params[
                                                                                 'DECODER_RNN_TYPE'] + 'Cond',
                                                                             trainable=self.trainable)

        rnn_output = sharedAttRNNCond(input_attentional_decoder)
        proj_h = rnn_output[0]
        x_att = rnn_output[1]
        alphas = rnn_output[2]
        h_state = rnn_output[3]

        trg_enc_frw = eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                     kernel_regularizer=l2(
                                                                         params['RECURRENT_WEIGHT_DECAY']),
                                                                     recurrent_regularizer=l2(
                                                                         params['RECURRENT_WEIGHT_DECAY']),
                                                                     bias_regularizer=l2(
                                                                         params['RECURRENT_WEIGHT_DECAY']),
                                                                     dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                                     recurrent_dropout=params[
                                                                         'RECURRENT_DROPOUT_P'],
                                                                     kernel_initializer=params['INIT_FUNCTION'],
                                                                     recurrent_initializer=params['INNER_INIT'],
                                                                     return_sequences=True,
                                                                     trainable=self.trainable,
                                                                     name='enc_trg_frw')

        trg_enc_bkw = eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                         kernel_regularizer=l2(
                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                         recurrent_regularizer=l2(
                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                         bias_regularizer=l2(
                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                         dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                         recurrent_dropout=params[
                                                             'RECURRENT_DROPOUT_P'],
                                                         kernel_initializer=params['INIT_FUNCTION'],
                                                         recurrent_initializer=params['INNER_INIT'],
                                                         return_sequences=True,
                                                         trainable=self.trainable,
                                                         go_backwards=True,
                                                         name='enc_trg_bkw')

        trg_state_frw = trg_enc_frw(state_below)
        trg_state_bkw = trg_enc_bkw(state_above)

        trg_state_bkw = Reverse(trg_state_bkw._keras_shape[2], axes=1, trainable=self.trainable, name='reverse_trg_state_bkw')(trg_state_bkw)

        # preparing formula 3b
        merged_emb = concatenate([state_below, state_above], axis=2, trainable=self.trainable, name='merged_emb')
        merged_states = concatenate([trg_state_frw, trg_state_bkw], axis=2, trainable=self.trainable, name='merged_states')

        # we replace state before with the concatenation of state before and after
        proj_h = merged_states

        if 'LSTM' in params['DECODER_RNN_TYPE']:
            h_memory = rnn_output[4]
        shared_Lambda_Permute = PermuteGeneral((1, 0, 2),trainable=self.trainable)

        if params['DOUBLE_STOCHASTIC_ATTENTION_REG'] > 0:
            alpha_regularizer = AlphaRegularizer(alpha_factor=params['DOUBLE_STOCHASTIC_ATTENTION_REG'])(alphas)

        [proj_h, shared_reg_proj_h] = Regularize(proj_h, params, trainable=self.trainable, shared_layers=True, name='proj_h0')

        # 3.4. Possibly deep decoder
        shared_proj_h_list = []
        shared_reg_proj_h_list = []

        h_states_list = [h_state]
        if 'LSTM' in params['DECODER_RNN_TYPE']:
            h_memories_list = [h_memory]

        for n_layer in range(1, params['N_LAYERS_DECODER']):
            current_rnn_input = [merged_states, shared_Lambda_Permute(x_att), initial_state]
            shared_proj_h_list.append(eval(params['DECODER_RNN_TYPE'].replace('Conditional', '') + 'Cond')(
                params['DECODER_HIDDEN_SIZE'],
                kernel_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                recurrent_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                conditional_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                bias_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                dropout=params['RECURRENT_DROPOUT_P'],
                recurrent_dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                conditional_dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                kernel_initializer=params['INIT_FUNCTION'],
                recurrent_initializer=params['INNER_INIT'],
                return_sequences=True,
                return_states=True,
                num_inputs=len(current_rnn_input),
                trainable=self.trainable,
                name='decoder_' + params['DECODER_RNN_TYPE'].replace(
                    'Conditional', '') + 'Cond' + str(n_layer)))

            if 'LSTM' in params['DECODER_RNN_TYPE']:
                current_rnn_input.append(initial_memory)
            current_rnn_output = shared_proj_h_list[-1](current_rnn_input)
            current_proj_h = current_rnn_output[0]
            h_states_list.append(current_rnn_output[1])
            if 'LSTM' in params['DECODER_RNN_TYPE']:
                h_memories_list.append(current_rnn_output[2])
            [current_proj_h, shared_reg_proj_h] = Regularize(current_proj_h, params, trainable=self.trainable, shared_layers=True,
                                                             name='proj_h' + str(n_layer))
            shared_reg_proj_h_list.append(shared_reg_proj_h)

            proj_h = Add(trainable=self.trainable)([proj_h, current_proj_h])

        # 3.5. Skip connections between encoder and output layer
        shared_FC_mlp = TimeDistributed(Dense(params['SKIP_VECTORS_HIDDEN_SIZE'],
                                              kernel_initializer=params['INIT_FUNCTION'],
                                              kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                              bias_regularizer=l2(params['WEIGHT_DECAY']),
                                              activation='linear',
                                              trainable=self.trainable),
                                        name='logit_lstm')
        out_layer_mlp = shared_FC_mlp(proj_h)
        shared_FC_ctx = TimeDistributed(Dense(params['SKIP_VECTORS_HIDDEN_SIZE'],
                                              kernel_initializer=params['INIT_FUNCTION'],
                                              kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                              bias_regularizer=l2(params['WEIGHT_DECAY']),
                                              activation='linear',
                                              trainable=self.trainable),
                                        name='logit_ctx')
        out_layer_ctx = shared_FC_ctx(x_att)
        out_layer_ctx = shared_Lambda_Permute(out_layer_ctx)
        shared_FC_emb = TimeDistributed(Dense(params['SKIP_VECTORS_HIDDEN_SIZE'],
                                              kernel_initializer=params['INIT_FUNCTION'],
                                              kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                              bias_regularizer=l2(params['WEIGHT_DECAY']),
                                              activation='linear',
                                              trainable=self.trainable),
                                        name='logit_emb')
        out_layer_emb = shared_FC_emb(merged_emb)

        [out_layer_mlp, shared_reg_out_layer_mlp] = Regularize(out_layer_mlp, params,
                                                               shared_layers=True, trainable=self.trainable, name='out_layer_mlp')
        [out_layer_ctx, shared_reg_out_layer_ctx] = Regularize(out_layer_ctx, params,
                                                               shared_layers=True, trainable=self.trainable, name='out_layer_ctx')
        [out_layer_emb, shared_reg_out_layer_emb] = Regularize(out_layer_emb, params,
                                                               shared_layers=True, trainable=self.trainable, name='out_layer_emb')

        shared_additional_output_merge = eval(params['ADDITIONAL_OUTPUT_MERGE_MODE'])(name='additional_input',
                                                                                      trainable=self.trainable)
        # formula 3b addition
        additional_output = shared_additional_output_merge([out_layer_mlp, out_layer_ctx, out_layer_emb])
        shared_activation_tanh = Activation('tanh', trainable=self.trainable)

        out_layer = shared_activation_tanh(additional_output)

        shared_deep_list = []
        shared_reg_deep_list = []

        # 3.6 Optional deep ouput layer
        for i, (activation, dimension) in enumerate(params['DEEP_OUTPUT_LAYERS']):
            shared_deep_list.append(TimeDistributed(Dense(dimension, activation=activation,
                                                          kernel_initializer=params['INIT_FUNCTION'],
                                                          kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                                          bias_regularizer=l2(params['WEIGHT_DECAY']),
                                                          trainable=self.trainable),
                                                    name=activation + '_%d' % i))
            out_layer = shared_deep_list[-1](out_layer)
            [out_layer, shared_reg_out_layer] = Regularize(out_layer,
                                                           params, trainable=self.trainable, shared_layers=True,
                                                           name='out_layer_' + str(activation) + '_%d' % i)
            shared_reg_deep_list.append(shared_reg_out_layer)

        shared_QE_soft = TimeDistributed(Dense(params['QE_VECTOR_SIZE'],
                                               use_bias=False,
                                               kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                               bias_regularizer=l2(params['WEIGHT_DECAY']),
                                               name='QE'+params['CLASSIFIER_ACTIVATION'],
                                               trainable=self.trainable),
                                         name='QE'+self.ids_outputs[0])

        # 3.7. Output layer: Softmax
        shared_FC_soft = TimeDistributed(Dense(params['OUTPUT_VOCABULARY_SIZE'],
                                               use_bias=False,
                                               activation=params['CLASSIFIER_ACTIVATION'],
                                               kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                               bias_regularizer=l2(params['WEIGHT_DECAY']),
                                               name=params['CLASSIFIER_ACTIVATION'],
                                               trainable = self.trainable),
                                         name=self.ids_outputs[0])

        softoutQE = shared_QE_soft(out_layer)
        softout = shared_FC_soft(softoutQE)

        self.model = Model(inputs=[src_text, next_words, next_words_bkw], outputs=[softout])
        if params['DOUBLE_STOCHASTIC_ATTENTION_REG'] > 0.:
            self.model.add_loss(alpha_regularizer)


def slice2d(x, dim, index):
    return x[:, index * dim: dim * (index + 1),:]


def get_slices2d(x, n):
    dim = 1
    return [Lambda(slice2d, arguments={'dim': dim, 'index': i}, output_shape=lambda s: (s[0], dim, s[2]))(x) for i in range(n)]


def slice3d(x, dim, index):
    return x[:, :, index * dim: dim * (index + 1)]

def get_slices3d(x, n):
    dim = int(K.int_shape(x)[2] / n)
    return [Lambda(slice3d, arguments={'dim': dim, 'index': i}, output_shape=lambda s: (s[0], s[1], dim))(x) for i in range(n)]

def merge(x, params, dim):
    return Lambda(lambda x: K.stack(x,axis=1), output_shape=(params['MAX_OUTPUT_TEXT_LEN'], dim * 2))(x)

def one_hot(x, params):
    return Lambda(lambda x: K.one_hot(x, params['OUTPUT_VOCABULARY_SIZE']), output_shape=(None, params['OUTPUT_VOCABULARY_SIZE']))(x)

def get_last_state(x, params):
    a = x[:, -1, :]
    return Lambda(lambda x: x[:, -1, :], output_shape=(1, params['QE_VECTOR_SIZE']*2))(x)

def max(x, params):
    return Lambda(lambda x: K.max(x, axis=2), output_shape=(None,params['MAX_OUTPUT_TEXT_LEN'], params['MAX_OUTPUT_TEXT_LEN']))(x)

def concat_time_distributed(input):
    a = input[0]
    b = input[1]
    return K.concatenate([a, b], axis=2)



class ShiftedConcat(Layer):

    def __init__(self, output_dim, params, reverse=False, **kwargs):
        self.output_dim = output_dim
        self.supports_masking = True
        self.params = params
        self.reverse = reverse
        super(ShiftedConcat, self).__init__(**kwargs)

    def call(self, x):
        seq1 = x[0]
        seq2 = x[1]

        if self.reverse:
            seq2= Reverse(seq2._keras_shape[2],axes=1)(seq2)

        # slice outputs per word
        sliced1 = get_slices2d(seq1, self.params['MAX_OUTPUT_TEXT_LEN'])
        sliced2 = get_slices2d(seq2, self.params['MAX_OUTPUT_TEXT_LEN'])

        states_merged = []

        #a loop over words
        for i in range(len(sliced1)):
            #for the first word and last words zeroes
            state_before = MyZeroesLayer()(sliced1[i])
            state_after = MyZeroesLayer()(sliced1[i])

            if i != 0:
                state_before = sliced1[i - 1]

            # for the last word EOS merge two identic states and embeddings
            if i < len(sliced1) - 1:
                state_after = sliced2[i + 1]

            # concatenate trg states, target embeddings
            state_merged = concatenate([state_before, state_after], axis=2)
            state_merged = K.batch_flatten(state_merged)
            states_merged.append(state_merged)

        return merge(states_merged, self.params, self.output_dim)

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)

    def compute_mask(self, inputs, mask):
        if isinstance(mask, list):
            mask = mask[0]
        else:
            mask= None
        return mask


class DenseTranspose(Layer):

  def __init__(self, output_dim, other_layer, other_layer_name, **kwargs):
      self.output_dim = output_dim
      self.other_layer=other_layer
      self.other_layer_name = other_layer_name
      super(DenseTranspose, self).__init__(**kwargs)

  def call(self, x):
      # w = self.other_layer.get_layer(self.other_layer_name).layer.kernel
      w = self.other_layer.layer.kernel
      w_trans = K.transpose(w)
      return K.dot(x, w_trans)

  def compute_output_shape(self, input_shape):
      return (input_shape[0], input_shape[1], self.output_dim)


class Reverse(Layer):

    def __init__(self, output_dim, axes, **kwargs):
        self.output_dim = output_dim
        self.axes = axes
        self.supports_masking = True
        super(Reverse, self).__init__(**kwargs)

    def call(self, x):
        return K.reverse(x, axes=self.axes)

    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1],self.output_dim)

    def compute_mask(self, inputs, mask):
        if isinstance(mask, list):
            mask = mask[0]
        else:
            mask= None
        return mask


class MyZeroesLayer(Layer):

    def __init__(self, **kwargs):
        self.supports_masking = True
        super(MyZeroesLayer, self).__init__(**kwargs)

    def call(self, x):
        return K.zeros_like(x)

    def compute_output_shape(self, input_shape):
        return (input_shape)

    def compute_mask(self, inputs, mask):
        if isinstance(mask, list):
            mask = mask[0]
        else:
            mask= None
        return mask


class GeneralReshape(Layer):

    def __init__(self, output_dim, params, **kwargs):
        self.output_dim = output_dim
        self.params = params
        super(GeneralReshape, self).__init__(**kwargs)

    def call(self, x):
        if len(self.output_dim)==2:
            return K.reshape(x, (-1, self.params['MAX_INPUT_TEXT_LEN']))
        if len(self.output_dim)==3:
            return K.reshape(x, (-1, self.output_dim[1], self.output_dim[2]))
        if len(self.output_dim)==4:
            return K.reshape(x, (-1, self.output_dim[1], self.output_dim[2], self.output_dim[3]))

    def compute_output_shape(self, input_shape):
        return self.output_dim


def attention_3d_block(inputs, params, ext):
    '''
    simple attention: weights over time steps; as in https://github.com/philipperemy/keras-attention-mechanism
    '''
    # inputs.shape = (batch_size, time_steps, input_dim)
    TIME_STEPS = K.int_shape(inputs)[1]
    input_dim = K.int_shape(inputs)[2]

    a = Permute((2, 1))(inputs)
    a = Dense(TIME_STEPS, activation='softmax', name='soft_att' + ext)(a)
    a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction' + ext, output_shape=(TIME_STEPS,))(a)
    a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec' + ext)(a)

    output_attention_mul = multiply([inputs, a_probs], name='attention_mul' + ext)
    sum = Lambda(reduce_sum, mask_aware_mean_output_shape)
    output = sum(output_attention_mul)

    return output


def reduce_max(x):
    return K.max(x, axis=1, keepdims=False)


def reduce_sum(x):
    return K.sum(x, axis=1, keepdims=False)



class NonMasking(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(NonMasking, self).__init__(**kwargs)

    def build(self, input_shape):
        input_shape = input_shape

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        return x

    def compute_output_shape(self, input_shape):
        return input_shape


def mask_aware_mean(x):
    '''
    see: https://github.com/keras-team/keras/issues/1579
    '''
    # recreate the masks - all zero rows have been masked
    mask = K.not_equal(K.sum(K.abs(x), axis=2, keepdims=True), 0)
    # number of that rows are not all zeros
    n = K.sum(K.cast(mask, 'float32'), axis=1, keepdims=False)

    x_mean = K.sum(x, axis=1, keepdims=False)
    x_mean = x_mean / n

    return x_mean


def mask_aware_mean_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 3
    return (shape[0], shape[2])


def mask_aware_mean4d(x):
    '''
    see: https://github.com/keras-team/keras/issues/1579
    '''
    # recreate the masks - all zero rows have been masked
    mask = K.not_equal(K.sum(K.abs(x), axis=3, keepdims=True), 0)
    # number of that rows are not all zeros
    n = K.sum(K.cast(mask, 'float32'), axis=2, keepdims=False)

    x_mean = K.sum(x, axis=2, keepdims=False)
    x_mean = x_mean / n

    return x_mean

def sum4d(x):

    return K.sum(x, axis=2, keepdims=False)


def mask_aware_merge_output_shape4d(input_shape):
    shape = list(input_shape)
    assert len(shape) == 4
    return (shape[0], shape[1], shape[3])
