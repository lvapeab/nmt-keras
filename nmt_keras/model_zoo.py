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
from keras_wrapper.extra.regularize import Regularize


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
        logging.info("<<< Obtaining positional encodings of layer " + name + " >>>")
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
                 model_name=None, vocabularies=None, store_path=None, set_optimizer=True, clear_dirs=True):
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

        self.use_CuDNN = 'CuDNN' if K.backend() == 'tensorflow' and params.get('USE_CUDNN', True) else ''

        # Prepare source word embedding
        if params['SRC_PRETRAINED_VECTORS'] is not None:
            if self.verbose > 0:
                logging.info("<<< Loading pretrained word vectors from:  " + params['SRC_PRETRAINED_VECTORS'] + " >>>")
            src_word_vectors = np.load(os.path.join(params['SRC_PRETRAINED_VECTORS'])).item()
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
                logging.info("<<< Loading pretrained word vectors from: " + params['TRG_PRETRAINED_VECTORS'] + " >>>")
            trg_word_vectors = np.load(os.path.join(params['TRG_PRETRAINED_VECTORS'])).item()
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
                logging.info("<<< Loading model structure from file " + structure_path + " >>>")
            self.model = model_from_json(open(structure_path).read())
        else:
            # Build model from scratch
            if hasattr(self, model_type):
                if self.verbose > 0:
                    logging.info("<<< Building " + model_type + " Translation_Model >>>")
                eval('self.' + model_type + '(params)')
            else:
                raise Exception('Translation_Model model_type "' + model_type + '" is not implemented.')

        # Load weights from file
        if weights_path:
            if self.verbose > 0:
                logging.info("<<< Loading weights from file " + weights_path + " >>>")
            self.model.load_weights(weights_path)

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
            logging.warning('Gradient accumulate is only implemented for the Adam optimizer. Setting "ACCUMULATE_GRADIENTS" to 1.')
            self.params['ACCUMULATE_GRADIENTS'] = 1

        optimizer_str = '\t LR: ' + str(self.params.get('LR', 0.01)) + \
                        '\n\t LOSS: ' + str(self.params.get('LOSS', 'categorical_crossentropy'))

        if self.params.get('USE_TF_OPTIMIZER', False) and K.backend() == 'tensorflow':
            if self.params['OPTIMIZER'].lower() not in ['sgd', 'adagrad', 'adadelta', 'rmsprop', 'adam']:
                logging.warning('The optimizer %s is not natively implemented in Tensorflow. Using the Keras version.' % (str(self.params['OPTIMIZER'])))
            if self.params.get('LR_DECAY') is not None:
                logging.warning('The learning rate decay is not natively implemented in native Tensorflow optimizers. Using the Keras version.')
                self.params['USE_TF_OPTIMIZER'] = False
            if self.params.get('ACCUMULATE_GRADIENTS', 1) > 1:
                logging.warning('The gradient accumulation is not natively implemented in native Tensorflow optimizers. Using the Keras version.')
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
            else:
                logging.info('\tWARNING: The modification of the LR is not implemented for the chosen optimizer.')
                optimizer = eval(self.params['OPTIMIZER'])

            optimizer_str += '\n\t CLIP_C ' + str(self.params.get('CLIP_C', 0.)) + \
                             '\n\t CLIP_V ' + str(self.params.get('CLIP_V', 0.)) + \
                             '\n\t LR_OPTIMIZER_DECAY ' + str(self.params.get('LR_OPTIMIZER_DECAY', 0.0)) + \
                             '\n\t ACCUMULATE_GRADIENTS ' + str(self.params.get('ACCUMULATE_GRADIENTS', 1)) + '\n'
        if self.verbose > 0:
            logging.info("Preparing optimizer and compiling. Optimizer configuration: \n" + optimizer_str)

        if hasattr(self, 'multi_gpu_model') and self.multi_gpu_model is not None:
            model_to_compile = self.multi_gpu_model
        else:
            model_to_compile = self.model

        model_to_compile.compile(optimizer=optimizer,
                                 loss=self.params['LOSS'],
                                 metrics=self.params.get('KERAS_METRICS', []),
                                 loss_weights=self.params.get('LOSS_WEIGHTS', None),
                                 sample_weight_mode='temporal' if self.params['SAMPLE_WEIGHTS'] else None,
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

        self.model = Model(inputs=[src_text, next_words], outputs=softout)

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
        self.model_init = Model(inputs=model_init_input, outputs=model_init_output)

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
                                outputs=model_next_outputs)
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
        self.model = Model(inputs=[src_text, next_words], outputs=softout)

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

        # if self.return_alphas:
        #    model_init_output.append(alphas)
        self.model_init = Model(inputs=model_init_input, outputs=model_init_output)

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
                                outputs=model_next_outputs)

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
