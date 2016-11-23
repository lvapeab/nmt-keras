from keras.engine import Input
from keras.engine.topology import merge
from keras.layers import TimeDistributed, Bidirectional
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU, LSTMCond, AttLSTM, AttLSTMCond, AttGRUCond
from keras.layers.convolutional import ZeroPadding1D
from keras.layers.core import Dropout, Dense, Flatten, Activation, Lambda, MaxoutDense, MaskedMean
from keras.models import model_from_json, Sequential, Graph, Model
from keras.regularizers import l2, activity_l2
from keras.optimizers import Adam, RMSprop, Nadam, Adadelta
from keras import backend as K
from keras.regularizers import l2

from keras_wrapper.cnn_model import Model_Wrapper
from utils.regularize import Regularize
import numpy as np
import os
import logging
import shutil
import time


class TranslationModel(Model_Wrapper):
    
    def resumeTrainNet(self, ds, parameters, out_name=None):
        pass

    def __init__(self, params, type='Translation_Model', verbose=1, structure_path=None, weights_path=None,
                 model_name=None, vocabularies=None, store_path=None):
        """
            Translation_Model object constructor.

            :param params: all hyperparameters of the model.
            :param type: network name type (corresponds to any method defined in the section 'MODELS' of this class).
                         Only valid if 'structure_path' == None.
            :param verbose: set to 0 if you don't want the model to output informative messages
            :param structure_path: path to a Keras' model json file.
                                  If we speficy this parameter then 'type' will be only an informative parameter.
            :param weights_path: path to the pre-trained weights file (if None, then it will be randomly initialized)
            :param model_name: optional name given to the network
                               (if None, then it will be assigned to current time as its name)
            :param vocabularies: vocabularies used for GLOVE word embedding
            :param store_path: path to the folder where the temporal model packups will be stored
        """
        super(self.__class__, self).__init__(type=type, model_name=model_name,
                                             silence=verbose == 0, models_path=store_path, inheritance=True)

        self.__toprint = ['_model_type', 'name', 'model_path', 'verbose']

        self.verbose = verbose
        self._model_type = type
        self.params = params
        self.vocabularies = vocabularies

        # Sets the model name and prepares the folders for storing the models
        self.setName(model_name, store_path=store_path)

        # Prepare source GLOVE embedding
        if params['SOURCE_GLOVE_VECTORS'] is not None:
            if self.verbose > 0:
                logging.info("<<< Loading pretrained word vectors from file " + params['SOURCE_GLOVE_VECTORS'] + " >>>")
            self.src_word_vectors = np.load(os.path.join(params['SOURCE_GLOVE_VECTORS'])).item()
            self.src_embedding_weights = np.random.rand(params['INPUT_VOCABULARY_SIZE'],
                                                        params['SOURCE_TEXT_EMBEDDING_SIZE'])
            for word, index in self.vocabularies[self.ids_inputs[0]]['words2idx'].iteritems():
                if self.src_word_vectors.get(word) is not None:
                    self.src_embedding_weights[index, :] = self.src_word_vectors[word]
            self.src_embedding_weights = [self.src_embedding_weights]
            self.src_embedding_weights_trainable = params['SOURCE_GLOVE_VECTORS_TRAINABLE']
        else:
            self.src_embedding_weights = None
            self.src_embedding_weights_trainable = True

        if params['TARGET_GLOVE_VECTORS'] is not None:
            if self.verbose > 0:
                logging.info("<<< Loading pretrained word vectors from file " + params['TARGET_GLOVE_VECTORS'] + " >>>")
            self.trg_word_vectors = np.load(os.path.join(params['TARGET_GLOVE_VECTORS'])).item()
            self.trg_embedding_weights = np.random.rand(params['OUTPUT_VOCABULARY_SIZE'],
                                                        params['TARGET_TEXT_EMBEDDING_SIZE'])
            for word, index in self.vocabularies[self.ids_inputs[0]]['words2idx'].iteritems():
                if self.src_word_vectors.get(word) is not None:
                    self.trg_embedding_weights[index, :] = self.trg_word_vectors[word]
            self.trg_embedding_weights = [self.trg_embedding_weights]
            self.trg_embedding_weights_trainable = params['SOURCE_GLOVE_VECTORS_TRAINABLE']

        else:
            self.trg_embedding_weights = None
            self.trg_embedding_weights_trainable = True

        # Prepare model
        if structure_path:
            # Load a .json model
            if self.verbose > 0:
                logging.info("<<< Loading model structure from file " + structure_path + " >>>")
            self.model = model_from_json(open(structure_path).read())
        else:
            # Build model from scratch
            if hasattr(self, type):
                if self.verbose > 0:
                    logging.info("<<< Building " + type + " Translation_Model >>>")
                eval('self.' + type + '(params)')
            else:
                raise Exception('Translation_Model type "' + type + '" is not implemented.')

        # Load weights from file
        if weights_path:
            if self.verbose > 0:
                logging.info("<<< Loading weights from file " + weights_path + " >>>")
            self.model.load_weights(weights_path)

        # Print information of self
        if verbose > 0:
            print str(self)
            self.model.summary()

        self.setOptimizer()

    def setOptimizer(self, **kwargs):

        """
            Sets a new optimizer for the Translation_Model.
            :param **kwargs:
        """

        # compile differently depending if our model is 'Sequential' or 'Graph'
        if self.verbose > 0:
            logging.info("Preparing optimizer and compiling.")
        if self.params['OPTIMIZER'].lower() == 'adam':
            optimizer = Adam(lr=self.params['LR'], clipnorm=self.params['CLIP_C'])
        elif self.params['OPTIMIZER'].lower() == 'rmsprop':
            optimizer = RMSprop(lr=self.params['LR'], clipnorm=self.params['CLIP_C'])
        elif self.params['OPTIMIZER'].lower() == 'nadam':
            optimizer = Nadam(lr=self.params['LR'], clipnorm=self.params['CLIP_C'])
        elif self.params['OPTIMIZER'].lower() == 'adadelta':
            optimizer = Adadelta(lr=self.params['LR'], clipnorm=self.params['CLIP_C'])
        else:
            logging.info('\tWARNING: The modification of the LR is not implemented for the chosen optimizer.')
            optimizer = self.params['OPTIMIZER']
        self.model.compile(optimizer=optimizer, loss=self.params['LOSS'],
                           sample_weight_mode='temporal' if self.params['SAMPLE_WEIGHTS'] else None)

    def setName(self, model_name, store_path=None, clear_dirs=True, **kwargs):
        """
         Changes the name (identifier) of the Translation_Model instance.
        :param model_name:
        :param store_path:
        :param clear_dirs:
        :param kwargs:
        :return:
        """
        if model_name is None:
            self.name = time.strftime("%Y-%m-%d") + '_' + time.strftime("%X")
            create_dirs = False
        else:
            self.name = model_name
            create_dirs = True

        if store_path is None:
            self.model_path = 'Models/' + self.name
        else:
            self.model_path = store_path

        # Remove directories if existed
        if clear_dirs:
            if os.path.isdir(self.model_path):
                shutil.rmtree(self.model_path)

        # Create new ones
        if create_dirs:
            if not os.path.isdir(self.model_path):
                os.makedirs(self.model_path)

    def __str__(self):
        """
            Plot basic model information.
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
        obj_str += 'MODEL PARAMETERS:\n'
        obj_str += str(self.params)
        obj_str += '\n'
        obj_str += '-----------------------------------------------------------------------------------'

        return obj_str

    # ------------------------------------------------------- #
    #       PREDEFINED MODELS
    # ------------------------------------------------------- #

    def GroundHogModel(self, params):
        """
        Machine translation with:
            * BLSTM encoder
            * Attention mechansim on input sequence of annotations
            * Conditional LSTM for decoding
            * Feed forward layers:
                + Context projected to output
                + Last word projected to output
        :param params: Dictionary of parameters (see config.py)
        :return:
        """

        self.ids_inputs = params["INPUTS_IDS_MODEL"]
        self.ids_outputs = params["OUTPUTS_IDS_MODEL"]

        # Encoder
        src_text = Input(name=self.ids_inputs[0], batch_shape=tuple([None, None]), dtype='int32')  # Source text input
        src_embedding = Embedding(params['INPUT_VOCABULARY_SIZE'], params['SOURCE_TEXT_EMBEDDING_SIZE'],
                                  name='source_word_embedding', W_regularizer=l2(params['WEIGHT_DECAY']),
                                  trainable=self.src_embedding_weights_trainable,  weights=self.src_embedding_weights,
                                  mask_zero=True)(src_text)
        src_embedding = Regularize(src_embedding, params, name='src_embedding')


        annotations = Bidirectional(GRU(params['ENCODER_HIDDEN_SIZE'],
                                        W_regularizer=l2(params['WEIGHT_DECAY']),
                                        U_regularizer=l2(params['WEIGHT_DECAY']),
                                        b_regularizer=l2(params['WEIGHT_DECAY']),
                                        return_sequences=True),
                                    name='bidirectional_encoder',
                                    merge_mode='concat')(src_embedding)

        annotations = Regularize(annotations, params, name='annotations')

        # Decoder
        # Previously generated words as inputs for training
        next_words = Input(name=self.ids_inputs[1], batch_shape=tuple([None, None]), dtype='int32')
        state_below = Embedding(params['OUTPUT_VOCABULARY_SIZE'], params['TARGET_TEXT_EMBEDDING_SIZE'],
                                name='target_word_embedding', W_regularizer=l2(params['WEIGHT_DECAY']),
                                trainable=self.trg_embedding_weights_trainable, weights=self.trg_embedding_weights,
                                mask_zero=True)(next_words)
        state_below = Regularize(state_below, params, name='state_below')

        # RNN initialization perceptrons with ctx mean
        ctx_mean = MaskedMean()(annotations)
        if len(params['INIT_LAYERS']) > 0:
            for n_layer_init in range(len(params['INIT_LAYERS'])-1):
                ctx_mean = Dense(params['DECODER_HIDDEN_SIZE'], name='init_layer_%d' % n_layer_init,
                                 activation=params['INIT_LAYERS'][n_layer_init],
                                 W_regularizer=l2(params['WEIGHT_DECAY']))(ctx_mean)
                ctx_mean = Regularize(ctx_mean, params, name='ctx' + str(n_layer_init))

            initial_state = Dense(params['DECODER_HIDDEN_SIZE'], name='initial_state',
                                  activation=params['INIT_LAYERS'][-1],
                                  W_regularizer=l2(params['WEIGHT_DECAY']))(ctx_mean)
            initial_state = Regularize(initial_state, params, name='initial_state')
            input_attentional_decoder = [state_below, annotations, initial_state]
        else:
            input_attentional_decoder = [state_below, annotations]

        # Decoder
        sharedAttGRUCond = AttGRUCond(params['DECODER_HIDDEN_SIZE'],
                                      W_regularizer=l2(params['WEIGHT_DECAY']),
                                      U_regularizer=l2(params['WEIGHT_DECAY']),
                                      V_regularizer=l2(params['WEIGHT_DECAY']),
                                      b_regularizer=l2(params['WEIGHT_DECAY']),
                                      wa_regularizer=l2(params['WEIGHT_DECAY']),
                                      Wa_regularizer=l2(params['WEIGHT_DECAY']),
                                      Ua_regularizer=l2(params['WEIGHT_DECAY']),
                                      ba_regularizer=l2(params['WEIGHT_DECAY']),
                                      return_sequences=True,
                                      return_extra_variables=True,
                                      return_states=True)

        [proj_h, x_att, alphas, h_state] = sharedAttGRUCond(input_attentional_decoder)
        proj_h = Regularize(proj_h, params, name='proj_h0')

        shared_FC_mlp = TimeDistributed(Dense(params['TARGET_TEXT_EMBEDDING_SIZE'], activation='linear',
                                              W_regularizer=l2(params['WEIGHT_DECAY'])), name='logit_lstm')
        out_layer_mlp = shared_FC_mlp(proj_h)
        shared_FC_ctx = TimeDistributed(Dense(params['TARGET_TEXT_EMBEDDING_SIZE'], activation='linear',
                                              W_regularizer=l2(params['WEIGHT_DECAY'])), name='logit_ctx')
        out_layer_ctx = shared_FC_ctx(x_att)
        shared_Lambda_Permute = Lambda(lambda x: K.permute_dimensions(x, [1, 0, 2]))
        out_layer_ctx = shared_Lambda_Permute(out_layer_ctx)
        shared_FC_emb = TimeDistributed(Dense(params['TARGET_TEXT_EMBEDDING_SIZE'], activation='linear',
                                              W_regularizer=l2(params['WEIGHT_DECAY'])), name='logit_emb')
        out_layer_emb = shared_FC_emb(state_below)

        out_layer_mlp = Regularize(out_layer_mlp, params, name='out_layer_mlp')
        out_layer_ctx = Regularize(out_layer_ctx, params, name='out_layer_ctx')
        out_layer_emb = Regularize(out_layer_emb, params, name='out_layer_emb')

        additional_output = merge([out_layer_mlp, out_layer_ctx, out_layer_emb], mode='sum', name='additional_input')

        shared_activation_tanh = Activation('tanh')

        out_layer = shared_activation_tanh(additional_output)

        shared_deep_list = []
        # Optional deep ouput
        for i, (activation, dimension) in enumerate(params['DEEP_OUTPUT_LAYERS']):
            if activation.lower() == 'maxout':
                shared_deep_list.append(TimeDistributed(MaxoutDense(dimension,
                                                                    W_regularizer=l2(params['WEIGHT_DECAY'])),
                                                        name='maxout_%d'%i))
            else:
                shared_deep_list.append(TimeDistributed(Dense(dimension, activation=activation,
                                              W_regularizer=l2(params['WEIGHT_DECAY'])),
                                            name=activation+'_%d'%i))
            out_layer = Regularize(out_layer, params, name='out_layer'+str(activation))
            out_layer = shared_deep_list[-1](out_layer)

        # Softmax
        shared_FC_soft = TimeDistributed(Dense(params['OUTPUT_VOCABULARY_SIZE'],
                                               activation=params['CLASSIFIER_ACTIVATION'],
                                               name=params['CLASSIFIER_ACTIVATION'],
                                               W_regularizer=l2(params['WEIGHT_DECAY'])),
                                         name=self.ids_outputs[0])
        softout = shared_FC_soft(out_layer)

        self.model = Model(input=[src_text, next_words], output=softout)

        ##################################################################
        #                     BEAM SEARCH MODEL
        ##################################################################
        # Now that we have the basic training model ready, let's prepare the model for applying decoding
        # The beam-search model will include all the minimum required set of layers (decoder stage) which offer the
        # possibility to generate the next state in the sequence given a pre-processed input (encoder stage)
        if params['BEAM_SEARCH']:
            # First, we need a model that outputs the preprocessed input + initial h state
            # for applying the initial forward pass
            self.model_init = Model(input=[src_text, next_words], output=[softout, annotations, h_state])

            # Store inputs and outputs names for model_init
            self.ids_inputs_init = self.ids_inputs
            # first output must be the output probs.
            self.ids_outputs_init = self.ids_outputs + ['preprocessed_input', 'next_state']

            # Second, we need to build an additional model with the capability to have the following inputs:
            #   - preprocessed_input
            #   - prev_word
            #   - prev_state
            # and the following outputs:
            #   - softmax probabilities
            #   - next_state
            preprocessed_size = params['ENCODER_HIDDEN_SIZE']*2
            # Define inputs
            preprocessed_annotations = Input(name='preprocessed_input', shape=tuple([None, preprocessed_size]))
            prev_h_state = Input(name='prev_state', shape=tuple([params['DECODER_HIDDEN_SIZE']]))
            input_attentional_decoder = [state_below, preprocessed_annotations, prev_h_state]
            # Apply decoder
            [proj_h, x_att, alphas, h_state] = sharedAttGRUCond(input_attentional_decoder)

            out_layer_mlp = shared_FC_mlp(proj_h)
            out_layer_ctx = shared_FC_ctx(x_att)
            out_layer_ctx = shared_Lambda_Permute(out_layer_ctx)
            out_layer_emb = shared_FC_emb(state_below)

            additional_output = merge([out_layer_mlp, out_layer_ctx, out_layer_emb], mode='sum', name='additional_input')
            out_layer = shared_activation_tanh(additional_output)

            for l in shared_deep_list:
                out_layer = l(out_layer)
            # Softmax
            softout = shared_FC_soft(out_layer)
            self.model_next = Model(input=[next_words, preprocessed_annotations, prev_h_state],
                                    output=[softout, preprocessed_annotations, h_state])

            # Store inputs and outputs names for model_next
            # first input must be previous word
            self.ids_inputs_next = [self.ids_inputs[1]] + ['preprocessed_input', 'prev_state']
            # first output must be the output probs.
            self.ids_outputs_next = self.ids_outputs + ['preprocessed_input', 'next_state']

            # Input -> Output matchings from model_init to model_next and from model_next to model_next
            self.matchings_init_to_next = {'preprocessed_input': 'preprocessed_input',
                                           'next_state': 'prev_state'}
            self.matchings_next_to_next = {'preprocessed_input': 'preprocessed_input',
                                           'next_state': 'prev_state'}
