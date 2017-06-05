import logging
import os

from keras.layers import *
from keras.models import model_from_json, Model
from keras.optimizers import Adam, RMSprop, Nadam, Adadelta, SGD, Adagrad, Adamax
from keras.regularizers import l2
from keras_wrapper.cnn_model import Model_Wrapper
from keras_wrapper.extra.regularize import Regularize


class TranslationModel(Model_Wrapper):
    """
    Translation model class. Instance of the Model_Wrapper class (see staged_keras_wrapper).

    :param params: all hyperparams of the model.
    :param model_type: network name type (corresponds to any method defined in the section 'MODELS' of this class).
                 Only valid if 'structure_path' == None.
    :param verbose: set to 0 if you don't want the model to output informative messages
    :param structure_path: path to a Keras' model json file.
                          If we speficy this parameter then 'type' will be only an informative parameter.
    :param weights_path: path to the pre-trained weights file (if None, then it will be randomly initialized)
    :param model_name: optional name given to the network
                       (if None, then it will be assigned to current time as its name)
    :param vocabularies: vocabularies used for word embedding
    :param store_path: path to the folder where the temporal model packups will be stored
    :param set_optimizer: Compile optimizer or not.
    :param clear_dirs: Clean model directories or not.

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
        :param model_name: optional name given to the network
                           (if None, then it will be assigned to current time as its name)
        :param vocabularies: vocabularies used for word embedding
        :param store_path: path to the folder where the temporal model packups will be stored
        :param set_optimizer: Compile optimizer or not.
        :param clear_dirs: Clean model directories or not.

        """
        super(TranslationModel, self).__init__(type=model_type, model_name=model_name,
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

        # Prepare source word embedding
        if params['SRC_PRETRAINED_VECTORS'] is not None:
            if self.verbose > 0:
                logging.info("<<< Loading pretrained word vectors from:  " + params['SRC_PRETRAINED_VECTORS'] + " >>>")
            src_word_vectors = np.load(os.path.join(params['SRC_PRETRAINED_VECTORS'])).item()
            self.src_embedding_weights = np.random.rand(params['INPUT_VOCABULARY_SIZE'],
                                                        params['SOURCE_TEXT_EMBEDDING_SIZE'])
            for word, index in self.vocabularies[self.ids_inputs[0]]['words2idx'].iteritems():
                if src_word_vectors.get(word) is not None:
                    self.src_embedding_weights[index, :] = src_word_vectors[word]
            self.src_embedding_weights = [self.src_embedding_weights]
            self.src_embedding_weights_trainable = params['SRC_PRETRAINED_VECTORS_TRAINABLE']
            del src_word_vectors

        else:
            self.src_embedding_weights = None
            self.src_embedding_weights_trainable = True

        # Prepare target word embedding
        if params['TRG_PRETRAINED_VECTORS'] is not None:
            if self.verbose > 0:
                logging.info("<<< Loading pretrained word vectors from: " + params['TRG_PRETRAINED_VECTORS'] + " >>>")
            trg_word_vectors = np.load(os.path.join(params['TRG_PRETRAINED_VECTORS'])).item()
            self.trg_embedding_weights = np.random.rand(params['OUTPUT_VOCABULARY_SIZE'],
                                                        params['TARGET_TEXT_EMBEDDING_SIZE'])
            for word, index in self.vocabularies[self.ids_outputs[0]]['words2idx'].iteritems():
                if trg_word_vectors.get(word) is not None:
                    self.trg_embedding_weights[index, :] = trg_word_vectors[word]
            self.trg_embedding_weights = [self.trg_embedding_weights]
            self.trg_embedding_weights_trainable = params['TRG_PRETRAINED_VECTORS_TRAINABLE']
            del trg_word_vectors
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
            print str(self)
            self.model.summary()
        if set_optimizer:
            self.setOptimizer()

    def setParams(self, params):
        self.params = params

    def setOptimizer(self, **kwargs):
        """
        Sets and compiles a new optimizer for the Translation_Model.
        :param kwargs:
        :return:
        """
        # compile differently depending if our model is 'Sequential' or 'Graph'
        if self.verbose > 0:
            logging.info("Preparing optimizer: %s [LR: %s - LOSS: %s] and compiling." %
                         (str(self.params['OPTIMIZER']), str(self.params.get('LR', 0.01)),
                          str(self.params.get('LOSS', 'categorical_crossentropy'))))

        if self.params['OPTIMIZER'].lower() == 'sgd':
            optimizer = SGD(lr=self.params.get('LR', 0.01),
                            momentum=self.params.get('MOMENTUM', 0.0),
                            decay=self.params.get('LR_OPTIMIZER_DECAY', 0.0),
                            nesterov=self.params.get('NESTEROV_MOMENTUM', False),
                            clipnorm=self.params.get('CLIP_C', 0.),
                            clipvalue=self.params.get('CLIP_V', 0.), )

        elif self.params['OPTIMIZER'].lower() == 'rsmprop':
            optimizer = RMSprop(lr=self.params.get('LR', 0.001),
                                rho=self.params.get('RHO', 0.9),
                                decay=self.params.get('LR_OPTIMIZER_DECAY', 0.0),
                                clipnorm=self.params.get('CLIP_C', 0.),
                                clipvalue=self.params.get('CLIP_V', 0.))

        elif self.params['OPTIMIZER'].lower() == 'adagrad':
            optimizer = Adagrad(lr=self.params.get('LR', 0.01),
                                decay=self.params.get('LR_OPTIMIZER_DECAY', 0.0),
                                clipnorm=self.params.get('CLIP_C', 0.),
                                clipvalue=self.params.get('CLIP_V', 0.))

        elif self.params['OPTIMIZER'].lower() == 'adadelta':
            optimizer = Adadelta(lr=self.params.get('LR', 1.0),
                                 rho=self.params.get('RHO', 0.9),
                                 decay=self.params.get('LR_OPTIMIZER_DECAY', 0.0),
                                 clipnorm=self.params.get('CLIP_C', 0.),
                                 clipvalue=self.params.get('CLIP_V', 0.))

        elif self.params['OPTIMIZER'].lower() == 'adam':
            optimizer = Adam(lr=self.params.get('LR', 0.001),
                             beta_1=self.params.get('BETA_1', 0.9),
                             beta_2=self.params.get('BETA_2', 0.999),
                             decay=self.params.get('LR_OPTIMIZER_DECAY', 0.0),
                             clipnorm=self.params.get('CLIP_C', 0.),
                             clipvalue=self.params.get('CLIP_V', 0.))

        elif self.params['OPTIMIZER'].lower() == 'adamax':
            optimizer = Adamax(lr=self.params.get('LR', 0.002),
                               beta_1=self.params.get('BETA_1', 0.9),
                               beta_2=self.params.get('BETA_2', 0.999),
                               decay=self.params.get('LR_OPTIMIZER_DECAY', 0.0),
                               clipnorm=self.params.get('CLIP_C', 0.),
                               clipvalue=self.params.get('CLIP_V', 0.))

        elif self.params['OPTIMIZER'].lower() == 'nadam':
            optimizer = Nadam(lr=self.params.get('LR', 0.002),
                              beta_1=self.params.get('BETA_1', 0.9),
                              beta_2=self.params.get('BETA_2', 0.999),
                              schedule_decay=self.params.get('LR_OPTIMIZER_DECAY', 0.0),
                              clipnorm=self.params.get('CLIP_C', 0.),
                              clipvalue=self.params.get('CLIP_V', 0.))
        else:
            logging.info('\tWARNING: The modification of the LR is not implemented for the chosen optimizer.')
            optimizer = eval(self.params['OPTIMIZER'])
        self.model.compile(optimizer=optimizer, loss=self.params['LOSS'],
                           metrics=self.params.get('KERAS_METRICS', []),
                           sample_weight_mode='temporal' if self.params['SAMPLE_WEIGHTS'] else None)

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
        obj_str += 'MODEL params:\n'
        obj_str += str(self.params)
        obj_str += '\n'
        obj_str += '-----------------------------------------------------------------------------------'

        return obj_str

    # ------------------------------------------------------- #
    #       PREDEFINED MODELS
    # ------------------------------------------------------- #

    def GroundHogModel(self, params):
        """
        Neural machine translation with:
            * BRNN encoder
            * Attention mechansim on input sequence of annotations
            * Conditional RNN for decoding
            * Deep output layers:
            * Context projected to output
            * Last word projected to output
            * Possibly deep encoder/decoder

        See https://arxiv.org/abs/1409.0473 for an in-depth review of the model.

        :param params: Dictionary of params (see config.py)
        :return: None
        """

        # 1. Source text input
        src_text = Input(name=self.ids_inputs[0], batch_shape=tuple([None, None]), dtype='int32')
        # 2. Encoder
        # 2.1. Source word embedding
        src_embedding = Embedding(params['INPUT_VOCABULARY_SIZE'], params['SOURCE_TEXT_EMBEDDING_SIZE'],
                                  name='source_word_embedding',
                                  W_regularizer=l2(params['WEIGHT_DECAY']),
                                  init=params['INIT_FUNCTION'],
                                  trainable=self.src_embedding_weights_trainable, weights=self.src_embedding_weights,
                                  mask_zero=True)(src_text)
        src_embedding = Regularize(src_embedding, params, name='src_embedding')

        # 2.2. BRNN encoder (GRU/LSTM)
        if params['BIDIRECTIONAL_ENCODER']:
            annotations = Bidirectional(eval(params['RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                 W_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                                 U_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                                 b_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                                 dropout_W=params['RECURRENT_DROPOUT_P'] if params[
                                                                     'USE_RECURRENT_DROPOUT'] else None,
                                                                 dropout_U=params['RECURRENT_DROPOUT_P'] if params[
                                                                     'USE_RECURRENT_DROPOUT'] else None,
                                                                 init=params['INIT_FUNCTION'],
                                                                 return_sequences=True),
                                        name='bidirectional_encoder_' + params['RNN_TYPE'],
                                        merge_mode='concat')(src_embedding)
        else:
            annotations = eval(params['RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                   W_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                   U_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                   b_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                   dropout_W=params['RECURRENT_DROPOUT_P'] if params[
                                                       'USE_RECURRENT_DROPOUT'] else None,
                                                   dropout_U=params['RECURRENT_DROPOUT_P'] if params[
                                                       'USE_RECURRENT_DROPOUT'] else None,
                                                   return_sequences=True,
                                                   init=params['INIT_FUNCTION'],
                                                   name='encoder_' + params['RNN_TYPE'])(src_embedding)
        annotations = Regularize(annotations, params, name='annotations')
        # 2.3. Potentially deep encoder
        for n_layer in range(1, params['N_LAYERS_ENCODER']):
            current_annotations = Bidirectional(eval(params['RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                         W_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         U_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         b_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         dropout_W=params['RECURRENT_DROPOUT_P'] if
                                                                         params['USE_RECURRENT_DROPOUT'] else None,
                                                                         dropout_U=params['RECURRENT_DROPOUT_P'] if
                                                                         params['USE_RECURRENT_DROPOUT'] else None,
                                                                         init=params['INIT_FUNCTION'],
                                                                         return_sequences=True,
                                                                         ),
                                                merge_mode='concat',
                                                name='bidirectional_encoder_' + str(n_layer))(annotations)
            current_annotations = Regularize(current_annotations, params, name='annotations_' + str(n_layer))
            annotations = merge([annotations, current_annotations], mode='sum')

        # 3. Decoder
        # 3.1.1. Previously generated words as inputs for training -> Teacher forcing
        next_words = Input(name=self.ids_inputs[1], batch_shape=tuple([None, None]), dtype='int32')
        # 3.1.2. Target word embedding
        state_below = Embedding(params['OUTPUT_VOCABULARY_SIZE'], params['TARGET_TEXT_EMBEDDING_SIZE'],
                                name='target_word_embedding',
                                W_regularizer=l2(params['WEIGHT_DECAY']),
                                init=params['INIT_FUNCTION'],
                                trainable=self.trg_embedding_weights_trainable, weights=self.trg_embedding_weights,
                                mask_zero=True)(next_words)
        state_below = Regularize(state_below, params, name='state_below')

        # 3.2. Decoder's RNN initialization perceptrons with ctx mean
        ctx_mean = MaskedMean()(annotations)
        annotations = MaskLayer()(annotations)  # We may want the padded annotations

        if len(params['INIT_LAYERS']) > 0:
            for n_layer_init in range(len(params['INIT_LAYERS']) - 1):
                ctx_mean = Dense(params['DECODER_HIDDEN_SIZE'], name='init_layer_%d' % n_layer_init,
                                 init=params['INIT_FUNCTION'],
                                 W_regularizer=l2(params['WEIGHT_DECAY']),
                                 activation=params['INIT_LAYERS'][n_layer_init]
                                 )(ctx_mean)
                ctx_mean = Regularize(ctx_mean, params, name='ctx' + str(n_layer_init))

            initial_state = Dense(params['DECODER_HIDDEN_SIZE'], name='initial_state',
                                  init=params['INIT_FUNCTION'],
                                  W_regularizer=l2(params['WEIGHT_DECAY']),
                                  activation=params['INIT_LAYERS'][-1]
                                  )(ctx_mean)
            initial_state = Regularize(initial_state, params, name='initial_state')
            input_attentional_decoder = [state_below, annotations, initial_state]

            if params['RNN_TYPE'] == 'LSTM':
                initial_memory = Dense(params['DECODER_HIDDEN_SIZE'], name='initial_memory',
                                       init=params['INIT_FUNCTION'],
                                       W_regularizer=l2(params['WEIGHT_DECAY']),
                                       activation=params['INIT_LAYERS'][-1])(ctx_mean)
                initial_memory = Regularize(initial_memory, params, name='initial_memory')
                input_attentional_decoder.append(initial_memory)
        else:
            # Initialize to zeros vector
            input_attentional_decoder = [state_below, annotations]
            initial_state = ZeroesLayer(params['DECODER_HIDDEN_SIZE'])(ctx_mean)
            input_attentional_decoder.append(initial_state)
            if params['RNN_TYPE'] == 'LSTM':
                input_attentional_decoder.append(initial_state)


        # 3.3. Attentional decoder
        sharedAttRNNCond = eval('Att' + params['RNN_TYPE'] + 'Cond')(params['DECODER_HIDDEN_SIZE'],
                                                                     att_dim=params.get('ATTENTION_SIZE', 0),
                                                                     W_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                                     U_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                                     V_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                                     b_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                                     wa_regularizer=l2(params['WEIGHT_DECAY']),
                                                                     Wa_regularizer=l2(params['WEIGHT_DECAY']),
                                                                     Ua_regularizer=l2(params['WEIGHT_DECAY']),
                                                                     ba_regularizer=l2(params['WEIGHT_DECAY']),
                                                                     dropout_W=params['RECURRENT_INPUT_DROPOUT_P'] if
                                                                     params['USE_RECURRENT_INPUT_DROPOUT'] else None,
                                                                     dropout_U=params['RECURRENT_DROPOUT_P'] if
                                                                     params['USE_RECURRENT_DROPOUT'] else None,
                                                                     dropout_V=params['RECURRENT_INPUT_DROPOUT_P'] if
                                                                     params['USE_RECURRENT_INPUT_DROPOUT'] else None,
                                                                     dropout_wa=params['DROPOUT_P']
                                                                     if params['USE_DROPOUT'] else None,
                                                                     dropout_Wa=params['DROPOUT_P']
                                                                     if params['USE_DROPOUT'] else None,
                                                                     dropout_Ua=params['DROPOUT_P']
                                                                     if params['USE_DROPOUT'] else None,
                                                                     init=params['INIT_FUNCTION'],
                                                                     return_sequences=True,
                                                                     return_extra_variables=True,
                                                                     return_states=True,
                                                                     name='decoder_Att' + params['RNN_TYPE'] + 'Cond')

        rnn_output = sharedAttRNNCond(input_attentional_decoder)
        proj_h = rnn_output[0]
        x_att = rnn_output[1]
        alphas = rnn_output[2]
        h_state = rnn_output[3]
        if params['RNN_TYPE'] == 'LSTM':
            h_memory = rnn_output[4]
        shared_Lambda_Permute = PermuteGeneral((1, 0, 2))

        [proj_h, shared_reg_proj_h] = Regularize(proj_h, params, shared_layers=True, name='proj_h0')

        # 3.4. Possibly deep decoder
        shared_proj_h_list = []
        shared_reg_proj_h_list = []

        h_states_list = [h_state]
        if params['RNN_TYPE'] == 'LSTM':
            h_memories_list = [h_memory]

        for n_layer in range(1, params['N_LAYERS_DECODER']):
            shared_proj_h_list.append(eval(params['RNN_TYPE'] + 'Cond')(params['DECODER_HIDDEN_SIZE'],
                                                                     W_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                                     U_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                                     V_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                                     b_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                                     dropout_W=params['RECURRENT_DROPOUT_P'] if params[
                                                                         'USE_RECURRENT_DROPOUT'] else None,
                                                                     dropout_U=params['RECURRENT_DROPOUT_P'] if params[
                                                                         'USE_RECURRENT_DROPOUT'] else None,
                                                                     dropout_V=params['RECURRENT_DROPOUT_P'] if params[
                                                                         'USE_RECURRENT_DROPOUT'] else None,
                                                                     init=params['INIT_FUNCTION'],
                                                                     return_sequences=True,
                                                                     return_states=True,
                                                                     name='decoder_' + params['RNN_TYPE'] +
                                                                          'Cond' + str(n_layer)))

            current_rnn_input = [proj_h, shared_Lambda_Permute(x_att), initial_state]
            if params['RNN_TYPE'] == 'LSTM':
                current_rnn_input.append(initial_memory)
            current_rnn_output = shared_proj_h_list[-1](current_rnn_input)
            current_proj_h = current_rnn_output[0]
            h_states_list.append(current_rnn_output[1])
            if params['RNN_TYPE'] == 'LSTM':
                h_memories_list.append(current_rnn_output[2])
            [current_proj_h, shared_reg_proj_h] = Regularize(current_proj_h, params, shared_layers=True,
                                                             name='proj_h' + str(n_layer))
            shared_reg_proj_h_list.append(shared_reg_proj_h)

            proj_h = merge([proj_h, current_proj_h], mode='sum')

        # 3.5. Skip connections between encoder and output layer
        shared_FC_mlp = TimeDistributed(Dense(params['SKIP_VECTORS_HIDDEN_SIZE'],
                                              init=params['INIT_FUNCTION'],
                                              W_regularizer=l2(params['WEIGHT_DECAY']),
                                              activation='linear',
                                              ), name='logit_lstm')
        out_layer_mlp = shared_FC_mlp(proj_h)
        shared_FC_ctx = TimeDistributed(Dense(params['SKIP_VECTORS_HIDDEN_SIZE'],
                                              init=params['INIT_FUNCTION'],
                                              W_regularizer=l2(params['WEIGHT_DECAY']),
                                              activation='linear',
                                              ), name='logit_ctx')
        out_layer_ctx = shared_FC_ctx(x_att)
        out_layer_ctx = shared_Lambda_Permute(out_layer_ctx)
        shared_FC_emb = TimeDistributed(Dense(params['SKIP_VECTORS_HIDDEN_SIZE'],
                                              init=params['INIT_FUNCTION'],
                                              W_regularizer=l2(params['WEIGHT_DECAY']),
                                              activation='linear'),
                                        name='logit_emb')
        out_layer_emb = shared_FC_emb(state_below)

        [out_layer_mlp, shared_reg_out_layer_mlp] = Regularize(out_layer_mlp, params,
                                                               shared_layers=True, name='out_layer_mlp')
        [out_layer_ctx, shared_reg_out_layer_ctx] = Regularize(out_layer_ctx, params,
                                                               shared_layers=True, name='out_layer_ctx')
        [out_layer_emb, shared_reg_out_layer_emb] = Regularize(out_layer_emb, params,
                                                               shared_layers=True, name='out_layer_emb')

        additional_output = merge([out_layer_mlp, out_layer_ctx, out_layer_emb],
                                  mode=params['ADDITIONAL_OUTPUT_MERGE_MODE'], name='additional_input')
        shared_activation_tanh = Activation('tanh')

        out_layer = shared_activation_tanh(additional_output)

        shared_deep_list = []
        shared_reg_deep_list = []
        # 3.6 Optional deep ouput layer
        for i, (activation, dimension) in enumerate(params['DEEP_OUTPUT_LAYERS']):
            if activation.lower() == 'maxout':
                shared_deep_list.append(TimeDistributed(MaxoutDense(dimension,
                                                                    init=params['INIT_FUNCTION'],
                                                                    W_regularizer=l2(params['WEIGHT_DECAY'])),
                                                        name='maxout_%d' % i))
            else:
                shared_deep_list.append(TimeDistributed(Dense(dimension, activation=activation,
                                                              init=params['INIT_FUNCTION'],
                                                              W_regularizer=l2(params['WEIGHT_DECAY'])),
                                                        name=activation + '_%d' % i))
            out_layer = shared_deep_list[-1](out_layer)
            [out_layer, shared_reg_out_layer] = Regularize(out_layer,
                                                           params, shared_layers=True,
                                                           name='out_layer' + str(activation))
            shared_reg_deep_list.append(shared_reg_out_layer)

        # 3.7. Output layer: Softmax
        shared_FC_soft = TimeDistributed(Dense(params['OUTPUT_VOCABULARY_SIZE'],
                                               activation=params['CLASSIFIER_ACTIVATION'],
                                               W_regularizer=l2(params['WEIGHT_DECAY']),
                                               name=params['CLASSIFIER_ACTIVATION']
                                               ),
                                         name=self.ids_outputs[0])
        softout = shared_FC_soft(out_layer)

        self.model = Model(input=[src_text, next_words], output=softout)

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
        if params['RNN_TYPE'] == 'LSTM':
            model_init_output += h_memories_list
        if self.return_alphas:
            model_init_output.append(alphas)
        self.model_init = Model(input=model_init_input, output=model_init_output)

        # Store inputs and outputs names for model_init
        self.ids_inputs_init = self.ids_inputs
        ids_states_names = ['next_state_' + str(i) for i in range(len(h_states_list))]

        # first output must be the output probs.
        self.ids_outputs_init = self.ids_outputs + ['preprocessed_input'] + ids_states_names
        if params['RNN_TYPE'] == 'LSTM':
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
            params['BIDIRECTIONAL_ENCODER'] \
            else params['ENCODER_HIDDEN_SIZE']
        # Define inputs
        n_deep_decoder_layer_idx = 0
        preprocessed_annotations = Input(name='preprocessed_input', shape=tuple([None, preprocessed_size]))
        prev_h_states_list = [Input(name='prev_state_' + str(i),
                                    shape=tuple([params['DECODER_HIDDEN_SIZE']]))
                              for i in range(len(h_states_list))]

        input_attentional_decoder = [state_below, preprocessed_annotations,
                                     prev_h_states_list[n_deep_decoder_layer_idx]]

        if params['RNN_TYPE'] == 'LSTM':
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
        if params['RNN_TYPE'] == 'LSTM':
            h_memories_list = [rnn_output[4]]
        for reg in shared_reg_proj_h:
            proj_h = reg(proj_h)

        for (rnn_decoder_layer, proj_h_reg) in zip(shared_proj_h_list, shared_reg_proj_h_list):
            n_deep_decoder_layer_idx += 1
            input_rnn_decoder_layer = [proj_h, shared_Lambda_Permute(x_att),
                                       prev_h_states_list[n_deep_decoder_layer_idx]]
            if params['RNN_TYPE'] == 'LSTM':
                input_rnn_decoder_layer.append(prev_h_memories_list[n_deep_decoder_layer_idx])

            current_rnn_output = rnn_decoder_layer(input_rnn_decoder_layer)
            current_proj_h = current_rnn_output[0]
            h_states_list.append(current_rnn_output[1])  # h_state
            if params['RNN_TYPE'] == 'LSTM':
                h_memories_list.append(current_rnn_output[2])  # h_memory
            for reg in proj_h_reg:
                current_proj_h = reg(current_proj_h)
            proj_h = merge([proj_h, current_proj_h], mode='sum')
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

        additional_output = merge([out_layer_mlp, out_layer_ctx, out_layer_emb],
                                  mode=params['ADDITIONAL_OUTPUT_MERGE_MODE'], name='additional_input_model_next')
        out_layer = shared_activation_tanh(additional_output)

        for (deep_out_layer, reg_list) in zip(shared_deep_list, shared_reg_deep_list):
            out_layer = deep_out_layer(out_layer)
            for reg in reg_list:
                out_layer = reg(out_layer)

        # Softmax
        softout = shared_FC_soft(out_layer)
        model_next_inputs = [next_words, preprocessed_annotations] + prev_h_states_list
        model_next_outputs = [softout, preprocessed_annotations] + h_states_list
        if params['RNN_TYPE'] == 'LSTM':
            model_next_inputs += prev_h_memories_list
            model_next_outputs += h_memories_list

        if self.return_alphas:
            model_next_outputs.append(alphas)

        self.model_next = Model(input=model_next_inputs,
                                output=model_next_outputs)
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

        if params['RNN_TYPE'] == 'LSTM':
            for n_memory in range(len(prev_h_memories_list)):
                self.ids_inputs_next.append('prev_memory_' + str(n_memory))
                self.ids_outputs_next.append('next_memory_' + str(n_memory))
                self.matchings_init_to_next['next_memory_' + str(n_memory)] = 'prev_memory_' + str(n_memory)
                self.matchings_next_to_next['next_memory_' + str(n_memory)] = 'prev_memory_' + str(n_memory)
