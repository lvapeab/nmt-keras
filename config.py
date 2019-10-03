def load_parameters():
    """
    Loads the defined hyperparameters
    :return parameters: Dictionary of loaded parameters
    """

    # Input data params
    TASK_NAME = 'qe-2016'                           # Task name
    DATASET_NAME = TASK_NAME                        # Dataset name
    SRC_LAN = 'src'                                  # Language of the source text
    TRG_LAN = 'mt'                                  # Language of the target text
    DATA_ROOT_PATH = 'examples/%s/' % DATASET_NAME  # Path where data is stored

    # SRC_LAN or TRG_LAN will be added to the file names
    TEXT_FILES = {'train': 'train.',        # Data files
                  'val': 'dev.',
                  'test': 'test.'}

    # Dataset class parameters
    INPUTS_IDS_DATASET = ['source_text', 'target_text']     # Corresponding inputs of the dataset
    #OUTPUTS_IDS_DATASET_FULL = ['target_text', 'word_qe', 'sent_hter']                   # Corresponding outputs of the dataset
    OUTPUTS_IDS_DATASET = ['sent_qe']
    INPUTS_IDS_MODEL = ['source_text', 'target_text']       # Corresponding inputs of the built model
    #OUTPUTS_IDS_MODEL_FULL = ['target_text','word_qe', 'sent_hter']                     # Corresponding outputs of the built model
    OUTPUTS_IDS_MODEL = ['sent_qe']
    WORD_QE_CLASSES = 5
    PRED_SCORE='hter'

    # Evaluation params
    METRICS = ['qe_metrics']                            # Metric used for evaluating the model
    #KERAS_METRICS = ['pearson_corr', 'mae', 'rmse']
    EVAL_ON_SETS = ['val', 'test']                        # Possible values: 'train', 'val' and 'test' (external evaluator)
    NO_REF = False
    #EVAL_ON_SETS_KERAS = ['val']                       #  Possible values: 'train', 'val' and 'test' (Keras' evaluator). Untested.
    EVAL_ON_SETS_KERAS = []
    START_EVAL_ON_EPOCH = 1                      # First epoch to start the model evaluation
    EVAL_EACH_EPOCHS = True                       # Select whether evaluate between N epochs or N updates
    EVAL_EACH = 1                                 # Sets the evaluation frequency (epochs or updates)

    #PRED_VOCAB = '/Users/ive/Documents/nmt-keras/datasets/Dataset_EuTrans_esen.pkl'
    #PRED_WEIGHTS='/home/julia/postech-qe-reproduced/epoch_2_weights.h5'
    MULTI_TASK = False

    # Search parameters
    SAMPLING = 'max_likelihood'                   # Possible values: multinomial or max_likelihood (recommended)
    TEMPERATURE = 1                               # Multinomial sampling parameter
    BEAM_SEARCH = False                            # Switches on-off the beam search procedure
    BEAM_SIZE = 6                                 # Beam size (in case of BEAM_SEARCH == True)
    OPTIMIZED_SEARCH = True                       # Compute annotations only a single time per sample
    SEARCH_PRUNING = False                        # Apply pruning strategies to the beam search method.
    MAXLEN_GIVEN_X = True                         # Generate translations of similar length to the source sentences
    MAXLEN_GIVEN_X_FACTOR = 2                     # The hypotheses will have (as maximum) the number of words of the
                                                  # source sentence * LENGTH_Y_GIVEN_X_FACTOR
    MINLEN_GIVEN_X = True                         # Generate translations of similar length to the source sentences
    MINLEN_GIVEN_X_FACTOR = 3                     # The hypotheses will have (as minimum) the number of words of the
                                                  # source sentence / LENGTH_Y_GIVEN_X_FACTOR

    # Apply length and coverage decoding normalizations.
    # See Section 7 from Wu et al. (2016) (https://arxiv.org/abs/1609.08144)
    LENGTH_PENALTY = False                        # Apply length penalty
    LENGTH_NORM_FACTOR = 0.2                      # Length penalty factor
    COVERAGE_PENALTY = False                      # Apply source coverage penalty
    COVERAGE_NORM_FACTOR = 0.2                    # Coverage penalty factor

    # Alternative (simple) length normalization.
    NORMALIZE_SAMPLING = False                    # Normalize hypotheses scores according to their length:
    ALPHA_FACTOR = .6                             # Normalization according to |h|**ALPHA_FACTOR

    # Sampling params: Show some samples during training
    # SAMPLE_ON_SETS = ['train', 'val']             # Possible values: 'train', 'val' and 'test'
    # N_SAMPLES = 5                                 # Number of samples generated
    # START_SAMPLING_ON_EPOCH = 600                   # First epoch where to start the sampling counter
    # SAMPLE_EACH_UPDATES = 300                     # Sampling frequency (always in #updates)

    # Unknown words treatment
    POS_UNK = True                                # Enable POS_UNK strategy for unknown words
    HEURISTIC = 0                                 # Heuristic to follow:
                                                  #     0: Replace the UNK by the correspondingly aligned source
                                                  #     1: Replace the UNK by the translation (given by an external
                                                  #        dictionary) of the correspondingly aligned source
                                                  #     2: Replace the UNK by the translation (given by an external
                                                  #        dictionary) of the correspondingly aligned source only if it
                                                  #        starts with a lowercase. Otherwise, copies the source word.
    ALIGN_FROM_RAW = True                         # Align using the full vocabulary or the short_list

    # Source -- Target pkl mapping (used for heuristics 1--2). See utils/build_mapping_file.sh for further info.

    MAPPING = DATA_ROOT_PATH + '/mapping.%s_%s.pkl' % (SRC_LAN, TRG_LAN)


    # Word representation params
    TOKENIZATION_METHOD = 'tokenize_none'         # Select which tokenization we'll apply.
                                                  # See Dataset class (from stager_keras_wrapper) for more info.
    BPE_CODES_PATH = DATA_ROOT_PATH + '/training_codes.joint'    # If TOKENIZATION_METHOD = 'tokenize_bpe',
                                                  # sets the path to the learned BPE codes.
    DETOKENIZATION_METHOD = 'detokenize_none'     # Select which de-tokenization method we'll apply

    APPLY_DETOKENIZATION = False                  # Wheter we apply a detokenization method

    TOKENIZE_HYPOTHESES = True  		          # Whether we tokenize the hypotheses using the
                                                  # previously defined tokenization method
    TOKENIZE_REFERENCES = True                    # Whether we tokenize the references using the
                                                  # previously defined tokenization method

    # Input image parameters
    DATA_AUGMENTATION = False                     # Apply data augmentation on input data (still unimplemented for text inputs)

    # Text parameters
    FILL = 'end'                                  # Whether we pad the 'end' or the 'start' of the sentence with 0s
    PAD_ON_BATCH = False                           # Whether we take as many timesteps as the longest sequence of
                                                  # the batch or a fixed size (MAX_OUTPUT_TEXT_LEN)
    # Input text parameters
    INPUT_VOCABULARY_SIZE = 30000                     # Size of the input vocabulary. Set to 0 for using all,
                                                  # otherwise it will be truncated to these most frequent words.
    MIN_OCCURRENCES_INPUT_VOCAB = 0               # Minimum number of occurrences allowed for the words in the input vocabulary.
                                                  # Set to 0 for using them all.
    MAX_INPUT_TEXT_LEN = 70                       # Maximum length of the input sequence

    # Output text parameters
    OUTPUT_VOCABULARY_SIZE = 30000                    # Size of the input vocabulary. Set to 0 for using all,
                                                  # otherwise it will be truncated to these most frequent words.
    MIN_OCCURRENCES_OUTPUT_VOCAB = 0              # Minimum number of occurrences allowed for the words in the output vocabulary.
    MAX_OUTPUT_TEXT_LEN = 70                      # Maximum length of the output sequence
                                                  # set to 0 if we want to use the whole answer as a single class
    MAX_OUTPUT_TEXT_LEN_TEST = MAX_OUTPUT_TEXT_LEN * 3  # Maximum length of the output sequence during test time

    # Optimizer parameters (see model.compile() function)
    LOSS = ['mse']
    #LOSS = ['categorical_crossentropy', 'mse', 'categorical_crossentropy']
    #predictor activation
    CLASSIFIER_ACTIVATION = 'softmax'

    OPTIMIZER = 'Adadelta'                            # Optimizer
    LR = 1.0                                    # Learning rate. Recommended values - Adam 0.001 - Adadelta 1.0
    CLIP_C = 1.                                   # During training, clip L2 norm of gradients to this value (0. means deactivated)
    CLIP_V = 0.                                   # During training, clip absolute value of gradients to this value (0. means deactivated)
    SAMPLE_WEIGHTS = {'word_qe': {'BAD': 3}}                        # Select whether we use a weights matrix (mask) for the data outputs
    # Learning rate annealing
    LR_DECAY = None                               # Frequency (number of epochs or updates) between LR annealings. Set to None for not decay the learning rate
    LR_GAMMA = 0.8                                # Multiplier used for decreasing the LR
    LR_REDUCE_EACH_EPOCHS = False                 # Reduce each LR_DECAY number of epochs or updates
    LR_START_REDUCTION_ON_EPOCH = 0               # Epoch to start the reduction
    LR_REDUCER_TYPE = 'exponential'               # Function to reduce. 'linear' and 'exponential' implemented.
    LR_REDUCER_EXP_BASE = 0.5                     # Base for the exponential decay
    LR_HALF_LIFE = 5000                           # Factor for exponenital decay

    # Training parameters
    MAX_EPOCH = 500 # Stop when computed this number of epochs
    EPOCH_PER_UPDATE = 1
    EPOCH_PER_PRED = 5
    EPOCH_PER_EST_SENT = 10
    EPOCH_PER_EST_WORD = 10
    #BATCH_SIZE = 2                               # Size of each minibatch
    #to use on real data
    BATCH_SIZE = 50

    HOMOGENEOUS_BATCHES = False                   # Use batches with homogeneous output lengths (Dangerous!!)
    JOINT_BATCHES = 4                             # When using homogeneous batches, get this number of batches to sort
    PARALLEL_LOADERS = 1                          # Parallel data batch loaders
    EPOCHS_FOR_SAVE = 1                           # Number of epochs between model saves
    WRITE_VALID_SAMPLES = True                    # Write valid samples in file
    SAVE_EACH_EVALUATION = True                   # Save each time we evaluate the model

    # Early stop parameters
    EARLY_STOP = True                             # Turns on/off the early stop protocol
    PATIENCE = 5                        # We'll stop if the val STOP_METRIC does not improve after this
                                                  # number of evaluations

    # was used for NMT
    STOP_METRIC = 'pearson'                        # Metric for the stop

    # Model parameters
    # Perictor+Estimator
    MODEL_TYPE = 'EncSent'                 # Model to train. See model_zoo() for the supported architectures

    # only Predictor
    #MODEL_TYPE = 'Predictor'

    #parameters for Predictor
    ENCODER_RNN_TYPE = 'GRU'                     # Encoder's RNN unit type ('LSTM' and 'GRU' supported)
    DECODER_RNN_TYPE = 'ConditionalGRU'          # Decoder's RNN unit type
                                                  # ('LSTM', 'GRU', 'ConditionalLSTM' and 'ConditionalGRU' supported)
    # Initializers (see keras/initializations.py).
    INIT_FUNCTION = 'glorot_uniform'              # General initialization function for matrices.
    INNER_INIT = 'orthogonal'                     # Initialization function for inner RNN matrices.
    INIT_ATT = 'glorot_uniform'                   # Initialization function for attention mechism matrices

    SOURCE_TEXT_EMBEDDING_SIZE = 300              # Source language word embedding size.
    SRC_PRETRAINED_VECTORS = None                 # Path to pretrained vectors (e.g.: DATA_ROOT_PATH + '/DATA/word2vec.%s.npy' % SRC_LAN)
                                                  # Set to None if you don't want to use pretrained vectors.
                                                  # When using pretrained word embeddings. this parameter must match with the word embeddings size
    SRC_PRETRAINED_VECTORS_TRAINABLE = True       # Finetune or not the target word embedding vectors.

    TARGET_TEXT_EMBEDDING_SIZE = 300               # Source language word embedding size.
    TRG_PRETRAINED_VECTORS = None                 # Path to pretrained vectors. (e.g. DATA_ROOT_PATH + '/DATA/word2vec.%s.npy' % TRG_LAN)
                                                  # Set to None if you don't want to use pretrained vectors.
                                                  # When using pretrained word embeddings, the size of the pretrained word embeddings must match with the word embeddings size.
    TRG_PRETRAINED_VECTORS_TRAINABLE = True       # Finetune or not the target word embedding vectors.

    # Encoder configuration
    ENCODER_HIDDEN_SIZE = 50                      # For models with RNN encoder
    BIDIRECTIONAL_ENCODER = True                  # Use bidirectional encoder
    N_LAYERS_ENCODER = 1                          # Stack this number of encoding layers
    BIDIRECTIONAL_DEEP_ENCODER = True             # Use bidirectional encoder in all encoding layers

    # Decoder configuration
    DECODER_HIDDEN_SIZE = 500                      # For models with RNN decoder
    N_LAYERS_DECODER = 1                          # Stack this number of decoding layers.
    ADDITIONAL_OUTPUT_MERGE_MODE = 'Add'          # Merge mode for the skip-connections (see keras.layers.merge.py)
    ATTENTION_SIZE = DECODER_HIDDEN_SIZE 
    # Skip connections size
    SKIP_VECTORS_HIDDEN_SIZE = TARGET_TEXT_EMBEDDING_SIZE

    #QE vector config
    QE_VECTOR_SIZE = 75

    # Fully-Connected layers for initializing the first RNN state
    #       Here we should only specify the activation function of each layer
    #       (as they have a potentially fixed size)
    #       (e.g INIT_LAYERS = ['tanh', 'relu'])
    INIT_LAYERS = ['tanh']

    # Additional Fully-Connected layers applied before softmax.
    #       Here we should specify the activation function and the output dimension
    #       (e.g DEEP_OUTPUT_LAYERS = [('tanh', 600), ('relu', 400), ('relu', 200)])
    DEEP_OUTPUT_LAYERS = [('linear', TARGET_TEXT_EMBEDDING_SIZE)]

    # Regularizers
    WEIGHT_DECAY = 1e-4                           # L2 regularization
    RECURRENT_WEIGHT_DECAY = 0.                   # L2 regularization in recurrent layers

    DROPOUT_P = 0.                                # Percentage of units to drop (0 means no dropout)
    RECURRENT_INPUT_DROPOUT_P = 0.                # Percentage of units to drop in input cells of recurrent layers
    RECURRENT_DROPOUT_P = 0.                      # Percentage of units to drop in recurrent layers

    USE_NOISE = True                              # Use gaussian noise during training
    NOISE_AMOUNT = 0.01                           # Amount of noise

    USE_BATCH_NORMALIZATION = True                # If True it is recommended to deactivate Dropout
    BATCH_NORMALIZATION_MODE = 1                  # See documentation in Keras' BN

    USE_PRELU = False                             # use PReLU activations as regularizer
    USE_L2 = False                                # L2 normalization on the features

    DOUBLE_STOCHASTIC_ATTENTION_REG = 0.0         # Doubly stochastic attention (Eq. 14 from arXiv:1502.03044)

    # Results plot and models storing parameters
    EXTRA_NAME = ''                               # This will be appended to the end of the model name
    MODEL_NAME = TASK_NAME + '_' + SRC_LAN + TRG_LAN + '_' + MODEL_TYPE
    #             '_src_emb_' + str(SOURCE_TEXT_EMBEDDING_SIZE) + \
    #             '_bidir_' + str(BIDIRECTIONAL_ENCODER) + \
    #             '_enc_' + ENCODER_RNN_TYPE + '_' + str(ENCODER_HIDDEN_SIZE) + \
    #             '_dec_' + DECODER_RNN_TYPE + '_' + str(DECODER_HIDDEN_SIZE) + \
    #             '_deepout_' + '_'.join([layer[0] for layer in DEEP_OUTPUT_LAYERS]) + \
    #             '_trg_emb_' + str(TARGET_TEXT_EMBEDDING_SIZE) + \
    #             '_' + OPTIMIZER + '_' + str(LR)

    MODEL_NAME += EXTRA_NAME

    STORE_PATH = 'trained_models/' + MODEL_NAME + '/'  # Models and evaluation results will be stored here
    DATASET_STORE_PATH = 'datasets/'                   # Dataset instance will be stored here

    SAMPLING_SAVE_MODE = 'list'                        # 'list': Store in a text file, one sentence per line.
    VERBOSE = 1                                        # Verbosity level
    RELOAD = 0                                         # If 0 start training from scratch, otherwise the model
                                                       # Saved on epoch 'RELOAD' will be used
    RELOAD_EPOCH = True                                # Select whether we reload epoch or update number

    REBUILD_DATASET = True                             # Build again or use stored instance
    MODE = 'training'                                  # 'training' or 'sampling' (if 'sampling' then RELOAD must
                                                       # be greater than 0 and EVAL_ON_SETS will be used)

    # Extra parameters for special trainings
    TRAIN_ON_TRAINVAL = False                          # train the model on both training and validation sets combined
    FORCE_RELOAD_VOCABULARY = False                    # force building a new vocabulary from the training samples
                                                       # applicable if RELOAD > 1

    # ================================================ #
    parameters = locals().copy()
    return parameters
