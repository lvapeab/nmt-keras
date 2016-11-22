from keras.engine import Input
from keras.engine.topology import merge
from keras.layers import TimeDistributed, Bidirectional
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU, LSTMCond, AttLSTM, AttLSTMCond, AttGRUCond
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization, L2_norm
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import ZeroPadding1D
from keras.layers.core import Dropout, Dense, Flatten, Activation, Lambda, MaxoutDense, MaskedMean
from keras.models import model_from_json, Sequential, Graph, Model
from keras.preprocessing.sequence import pad_sequences
from keras.regularizers import l2, activity_l2
from keras.layers.convolutional import AveragePooling1D
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
import sys
sys.setrecursionlimit(1500)

class Translation_Model(Model_Wrapper):
    
    def __init__(self, params, type='Basic_Multimodal_Translation_Model', verbose=1, structure_path=None, weights_path=None,
                 model_name=None, vocabularies=None, store_path=None):
        """
            Translation_Model object constructor.

            :param params: all hyperparameters of the model.
            :param type: network name type (corresponds to any method defined in the section 'MODELS' of this class). Only valid if 'structure_path' == None.
            :param verbose: set to 0 if you don't want the model to output informative messages
            :param structure_path: path to a Keras' model json file. If we speficy this parameter then 'type' will be only an informative parameter.
            :param weights_path: path to the pre-trained weights file (if None, then it will be randomly initialized)
            :param model_name: optional name given to the network (if None, then it will be assigned to current time as its name)
            :param vocabularies: vocabularies used for GLOVE word embedding
            :param store_path: path to the folder where the temporal model packups will be stored

            References:
                [PReLU]
                Kaiming He et al. Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification

                [BatchNormalization]
                Sergey Ioffe and Christian Szegedy. Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
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

        # Prepare GLOVE embedding
        if params['GLOVE_VECTORS'] is not None:
            if self.verbose > 0:
                logging.info("<<< Loading pretrained word vectors from file "+ params['GLOVE_VECTORS'] +" >>>")
            self.word_vectors = np.load(os.path.join(params['GLOVE_VECTORS'])).item()
        else:
            self.word_vectors = dict()

        # Prepare model
        if structure_path:
            # Load a .json model
            if self.verbose > 0:
                logging.info("<<< Loading model structure from file "+ structure_path +" >>>")
            self.model = model_from_json(open(structure_path).read())
        else:
            # Build model from scratch
            if hasattr(self, type):
                if self.verbose > 0:
                    logging.info("<<< Building "+ type +" Translation_Model >>>")
                eval('self.'+type+'(params)')
            else:
                raise Exception('Translation_Model type "'+ type +'" is not implemented.')

        # Load weights from file
        if weights_path:
            if self.verbose > 0:
                logging.info("<<< Loading weights from file "+ weights_path +" >>>")
            self.model.load_weights(weights_path)

        # Print information of self
        if verbose > 0:
            print str(self)
            self.model.summary()

        self.setOptimizer()

    def setOptimizer(self):

        """
            Sets a new optimizer for the Translation_Model.
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


    def setName(self, model_name, store_path=None, clear_dirs=True):
        """
            Changes the name (identifier) of the Translation_Model instance.
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

    # ------------------------------------------------------- #
    #       VISUALIZATION
    #           Methods for visualization
    # ------------------------------------------------------- #

    def __str__(self):
        """
            Plot basic model information.
        """
        obj_str = '-----------------------------------------------------------------------------------\n'
        class_name = self.__class__.__name__
        obj_str += '\t\t'+class_name +' instance\n'
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
        :param params:
        :return:
        """

        self.ids_inputs = params["INPUTS_IDS_MODEL"]
        self.ids_outputs = params["OUTPUTS_IDS_MODEL"]


        # Source text
        src_text = Input(name=self.ids_inputs[0], batch_shape=tuple([None, None]), dtype='int32')
        src_embedding = Embedding(params['INPUT_VOCABULARY_SIZE'], params['TEXT_EMBEDDING_HIDDEN_SIZE'],
                        name='source_word_embedding',
                        W_regularizer=l2(params['WEIGHT_DECAY']),
                        mask_zero=True)(src_text)
        src_embedding = Regularize(src_embedding, params, name='src_embedding')
        # Previously generated words as inputs for training
        next_words = Input(name=self.ids_inputs[1], batch_shape=tuple([None, None]), dtype='int32')
        state_below = Embedding(params['OUTPUT_VOCABULARY_SIZE'], params['TEXT_EMBEDDING_HIDDEN_SIZE'],
                        name='target_word_embedding',
                        W_regularizer=l2(params['WEIGHT_DECAY']),
                        mask_zero=True)(next_words)
        state_below = Regularize(state_below, params, name='state_below')

        annotations = Bidirectional(GRU(params['ENCODER_HIDDEN_SIZE'],
                                             W_regularizer=l2(params['WEIGHT_DECAY']),
                                             U_regularizer=l2(params['WEIGHT_DECAY']),
                                             b_regularizer=l2(params['WEIGHT_DECAY']),
                                             return_sequences=True),
                                        name='bidirectional_encoder', merge_mode='concat')(src_embedding)
        annotations = Regularize(annotations, params, name='annotations')

        # LSTM initialization perceptrons with ctx mean
        ctx_mean = MaskedMean()(annotations)
        if len(params['INIT_LAYERS']) > 0:
            for n_layer_init in range(len(params['INIT_LAYERS'])-1):
                ctx_mean = Dense(params['DECODER_HIDDEN_SIZE'], name='init_layer_%d'%n_layer_init,
                                 activation=params['INIT_LAYERS'][n_layer_init],
                                 W_regularizer=l2(params['WEIGHT_DECAY']))(ctx_mean)
                ctx_mean = Regularize(ctx_mean, params, name='ctx' + str(n_layer_init))

            initial_state = Dense(params['DECODER_HIDDEN_SIZE'], name='initial_state',
                                  activation=params['INIT_LAYERS'][-1],
                                  W_regularizer=l2(params['WEIGHT_DECAY']))(ctx_mean)
            initial_state =  Regularize(initial_state, params, name='initial_state')
            input_attlstmcond = [state_below, annotations, initial_state]
        else:
            input_attlstmcond = [state_below, annotations]

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


        [proj_h, x_att, alphas, h_state] = sharedAttGRUCond(input_attlstmcond)
        proj_h= Regularize(proj_h, params,  name='proj_h0')

        shared_FC_mlp = TimeDistributed(Dense(params['TEXT_EMBEDDING_HIDDEN_SIZE'], activation='linear',
                                              W_regularizer=l2(params['WEIGHT_DECAY'])), name='logit_lstm')
        out_layer_mlp = shared_FC_mlp(proj_h)
        shared_FC_ctx = TimeDistributed(Dense(params['TEXT_EMBEDDING_HIDDEN_SIZE'], activation='linear',
                                              W_regularizer=l2(params['WEIGHT_DECAY'])), name='logit_ctx')
        out_layer_ctx = shared_FC_ctx(x_att)
        shared_Lambda_Permute = Lambda(lambda x: K.permute_dimensions(x, [1, 0, 2]))
        out_layer_ctx = shared_Lambda_Permute(out_layer_ctx)
        shared_FC_emb = TimeDistributed(Dense(params['TEXT_EMBEDDING_HIDDEN_SIZE'], activation='linear',
                                              W_regularizer=l2(params['WEIGHT_DECAY'])), name='logit_emb')
        out_layer_emb = shared_FC_emb(state_below)

        out_layer_mlp = Regularize(out_layer_mlp, params, name='out_layer_mlp')
        out_layer_ctx = Regularize(out_layer_ctx, params, name='out_layer_ctx')
        out_layer_emb = Regularize(out_layer_emb, params, name='out_layer_emb')

        additional_output = merge([out_layer_mlp, out_layer_ctx, out_layer_emb], mode='sum', name='additional_input')

        shared_activation_tanh = Activation('tanh')

        out_layer = shared_activation_tanh(additional_output)

        shared_Deep_list = []
        # Optional deep ouput
        for i, (activation, dimension) in enumerate(params['DEEP_OUTPUT_LAYERS']):
            if activation.lower() == 'maxout':
                shared_Deep_list.append(TimeDistributed(MaxoutDense(dimension,
                                                                    W_regularizer=l2(params['WEIGHT_DECAY'])),
                                                        name='maxout_%d'%i))
            else:
                shared_Deep_list.append(TimeDistributed(Dense(dimension, activation=activation,
                                              W_regularizer=l2(params['WEIGHT_DECAY'])),
                                            name=activation+'_%d'%i))
            out_layer = Regularize(out_layer, params, name='out_layer'+str(activation))
            out_layer = shared_Deep_list[-1](out_layer)

        # Softmax
        shared_FC_soft = TimeDistributed(Dense(params['OUTPUT_VOCABULARY_SIZE'], activation=params['CLASSIFIER_ACTIVATION'],
                                       name=params['CLASSIFIER_ACTIVATION'],
                                       W_regularizer=l2(params['WEIGHT_DECAY'])),
                                 name=self.ids_outputs[0])
        softout = shared_FC_soft(out_layer)

        self.model = Model(input=[src_text, next_words], output=softout)

        ##################################################################
        #                     BEAM SEARCH MODEL
        ##################################################################
        # Now that we have the basic training model ready, let's prepare the model for applying a beam search during sampling
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
            input_attlstmcond = [state_below, preprocessed_annotations, prev_h_state]
            # Apply decoder
            [proj_h, x_att, alphas, h_state] = sharedAttGRUCond(input_attlstmcond)

            out_layer_mlp = shared_FC_mlp(proj_h)
            out_layer_ctx = shared_FC_ctx(x_att)
            out_layer_ctx = shared_Lambda_Permute(out_layer_ctx)
            out_layer_emb = shared_FC_emb(state_below)

            additional_output = merge([out_layer_mlp, out_layer_ctx, out_layer_emb], mode='sum', name='additional_input')
            out_layer = shared_activation_tanh(additional_output)

            for l in shared_Deep_list:
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


    def GroundHogModelOldSearch(self, params):
        """
        Machine translation with:
            * BLSTM encoder
            * Attention mechansim on input sequence of annotations
            * Conditional LSTM for decoding
            * Feed forward layers:
                + Context projected to output
                + Last word projected to output
        :param params:
        :return:
        """


        # Store inputs and outputs names
        self.ids_inputs =  params['INPUTS_IDS_MODEL']
        self.ids_outputs = params['OUTPUTS_IDS_MODEL']

        # Source text
        src_text = Input(name=self.ids_inputs[0], batch_shape=tuple([None, None]), dtype='int32')
        src_embedding = Embedding(params['INPUT_VOCABULARY_SIZE'], params['TEXT_EMBEDDING_HIDDEN_SIZE'],
                        name='source_word_embedding',
                        W_regularizer=l2(params['WEIGHT_DECAY']),
                        mask_zero=True)(src_text)
        src_embedding = Regularize(src_embedding, params, name='src_embedding')
        # Previously generated words as inputs for training
        next_words = Input(name=self.ids_inputs[1], batch_shape=tuple([None, None]), dtype='int32')
        state_below = Embedding(params['OUTPUT_VOCABULARY_SIZE'], params['TEXT_EMBEDDING_HIDDEN_SIZE'],
                        name='target_word_embedding',
                        W_regularizer=l2(params['WEIGHT_DECAY']),
                        mask_zero=True)(next_words)
        state_below = Regularize(state_below, params, name='state_below')

        annotations = Bidirectional(GRU(params['ENCODER_HIDDEN_SIZE'],
                                             W_regularizer=l2(params['WEIGHT_DECAY']),
                                             U_regularizer=l2(params['WEIGHT_DECAY']),
                                             b_regularizer=l2(params['WEIGHT_DECAY']),
                                             return_sequences=True),
                                        name='bidirectional_encoder')(src_embedding)

        # LSTM initialization perceptrons with ctx mean
        ctx_mean = MaskedMean()(annotations)
        ctx_mean = Regularize(ctx_mean, params, name='ctx_mean')(ctx_mean)

        if len(params['INIT_LAYERS']) > 0:
            for n_layer_init in range(len(params['INIT_LAYERS'])-1):
                ctx_mean = Dense(params['DECODER_HIDDEN_SIZE'], name='init_layer_%d'%nlayer_init,
                                 activation=params['INIT_LAYERS'][nlayer_init],
                                 W_regularizer=l2(params['WEIGHT_DECAY']))(ctx_mean)
                ctx_mean = Regularize(ctx_mean, params, name='ctx' + str(n_layer_init))

            initial_state = Dense(params['DECODER_HIDDEN_SIZE'], name='initial_state',
                                  activation=params['INIT_LAYERS'][-1],
                                  W_regularizer=l2(params['WEIGHT_DECAY']))(ctx_mean)
            initial_state =  Regularize(initial_state, params, name='initial_state')
            input_attlstmcond = [state_below, annotations, initial_state]
        else:
            input_attlstmcond = [state_below, annotations]

        # x_att
        [proj_h, x_att, alphas] = AttGRUCond(params['DECODER_HIDDEN_SIZE'],
                                              W_regularizer=l2(params['WEIGHT_DECAY']),
                                              U_regularizer=l2(params['WEIGHT_DECAY']),
                                              V_regularizer=l2(params['WEIGHT_DECAY']),
                                              b_regularizer=l2(params['WEIGHT_DECAY']),
                                              wa_regularizer=l2(params['WEIGHT_DECAY']),
                                              Wa_regularizer=l2(params['WEIGHT_DECAY']),
                                              Ua_regularizer=l2(params['WEIGHT_DECAY']),
                                              ba_regularizer=l2(params['WEIGHT_DECAY']),
                                              init='norm_weight',
                                              inner_init='ortho_weight',
                                              return_sequences=True,
                                              return_extra_variables=True)(input_attlstmcond)
        prev_proj_h = Regularize(proj_h, params, name='proj_h0')


        for n_layer_lstm in range(1, params['N_LAYERS_DECODER_LSTM']):
            current_proj_h = GRU(params['LSTM_DECODER_HIDDEN_SIZE'],
                                              W_regularizer=l2(params['WEIGHT_DECAY']),
                                              U_regularizer=l2(params['WEIGHT_DECAY']),
                                              b_regularizer=l2(params['WEIGHT_DECAY']),
                                              return_sequences=True)(prev_proj_h)
            current_proj_h = Regularize(current_proj_h, params, name='proj_h' + str(n_layer_lstm))
            prev_proj_h = merge([prev_proj_h, current_proj_h], mode='sum')
        proj_h = prev_proj_h



        # Equation 7 from Show, attend and tell (http://arxiv.org/abs/1502.03044)
        out_layer_mlp = TimeDistributed(Dense(params['TEXT_EMBEDDING_HIDDEN_SIZE'], activation='linear',
                                              W_regularizer=l2(params['WEIGHT_DECAY'])), name='logit_lstm')(proj_h)
        out_layer_ctx = TimeDistributed(Dense(params['TEXT_EMBEDDING_HIDDEN_SIZE'], activation='linear',
                                              W_regularizer=l2(params['WEIGHT_DECAY'])), name='logit_ctx')(x_att)
        out_layer_ctx = Lambda(lambda x: K.permute_dimensions(x, [1, 0, 2]))(out_layer_ctx)

        out_layer_emb = TimeDistributed(Dense(params['TEXT_EMBEDDING_HIDDEN_SIZE'], activation='linear',
                                              W_regularizer=l2(params['WEIGHT_DECAY'])), name='logit_emb')(state_below)

        out_layer_mlp = Regularize(out_layer_mlp, params, name='out_layer_mlp')
        out_layer_ctx = Regularize(out_layer_ctx, params, name='out_layer_ctx')
        out_layer_emb = Regularize(out_layer_emb, params, name='out_layer_emb')
        additional_output = merge([out_layer_mlp, out_layer_ctx, out_layer_emb], mode='sum', name='additional_input')
        out_layer = Activation('tanh')(additional_output)

        # Optional deep ouput
        for i, (activation, dimension) in enumerate(params['DEEP_OUTPUT_LAYERS']):
            if activation.lower() == 'maxout':
                out_layer = TimeDistributed(MaxoutDense(dimension, W_regularizer=l2(params['WEIGHT_DECAY'])),
                                            name='maxout_%d'%i)(out_layer)
            else:
                out_layer = TimeDistributed(Dense(dimension, activation=activation,
                                              W_regularizer=l2(params['WEIGHT_DECAY'])),
                                            name=activation+'_%d'%i)(out_layer)
            out_layer = Regularize(out_layer, params, name='out_layer'+str(activation))

        # Softmax
        output = TimeDistributed(Dense(params['OUTPUT_VOCABULARY_SIZE'], activation=params['CLASSIFIER_ACTIVATION'],
                                       name=params['CLASSIFIER_ACTIVATION'], W_regularizer=l2(params['WEIGHT_DECAY'])),
                                 name=self.ids_outputs[0])(out_layer)

        self.model = Model(input=[src_text, next_words], output=output)
