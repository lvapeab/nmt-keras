
def load_parameters():
    """
    Loads the defined hyperparameters
    :return parameters: Dictionary of loaded parameters
    """

    # Input data params
    TASK_NAME = 'EuTrans'                           # Task name
    DATASET_NAME = TASK_NAME                        # Dataset name
    SRC_LAN = 'es'                                  # Language of the source text
    TRG_LAN = 'en'                                  # Language of the target text
    DATA_ROOT_PATH = 'examples/%s/' % DATASET_NAME  # Path where data is stored

    # SRC_LAN or TRG_LAN will be added to the file names
    TEXT_FILES = {'train': 'training.',        # Data files
                  'val': 'dev.',
                  'test': 'test.'}

    # Dataset class parameters
    INPUTS_IDS_DATASET = ['source_text', 'state_below']     # Corresponding inputs of the dataset
    OUTPUTS_IDS_DATASET = ['target_text']                   # Corresponding outputs of the dataset
    INPUTS_IDS_MODEL = ['source_text', 'state_below']       # Corresponding inputs of the built model
    OUTPUTS_IDS_MODEL = ['target_text']                     # Corresponding outputs of the built model

    # Evaluation params
    METRICS = ['coco']                            # Metric used for evaluating the model
    EVAL_ON_SETS = ['val']                        # Possible values: 'train', 'val' and 'test' (external evaluator)
    EVAL_ON_SETS_KERAS = []                       # Possible values: 'train', 'val' and 'test' (Keras' evaluator). Untested.
    START_EVAL_ON_EPOCH = 1                       # First epoch to start the model evaluation
    EVAL_EACH_EPOCHS = True                       # Select whether evaluate between N epochs or N updates
    EVAL_EACH = 1                                 # Sets the evaluation frequency (epochs or updates)

    # Search parameters
    SAMPLING = 'max_likelihood'                   # Possible values: multinomial or max_likelihood (recommended)
    TEMPERATURE = 1                               # Multinomial sampling parameter
    BEAM_SEARCH = True                            # Switches on-off the beam search procedure
    BEAM_SIZE = 6                                 # Beam size (in case of BEAM_SEARCH == True)
    OPTIMIZED_SEARCH = True                       # Compute annotations only a single time per sample
    SEARCH_PRUNING = False                        # Apply pruning strategies to the beam search method.
                                                  # It will likely increase decoding speed, but decrease quality.

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
    SAMPLE_ON_SETS = ['train', 'val']             # Possible values: 'train', 'val' and 'test'
    N_SAMPLES = 5                                 # Number of samples generated
    START_SAMPLING_ON_EPOCH = 1                   # First epoch where to start the sampling counter
    SAMPLE_EACH_UPDATES = 300                     # Sampling frequency (always in #updates)

    # Unknown words treatment
    POS_UNK = False                               # Enable POS_UNK strategy for unknown words
    HEURISTIC = 0                                 # Heuristic to follow:
                                                  #     0: Replace the UNK by the correspondingly aligned source
                                                  #     1: Replace the UNK by the translation (given by an external
                                                  #        dictionary) of the correspondingly aligned source
                                                  #     2: Replace the UNK by the translation (given by an external
                                                  #        dictionary) of the correspondingly aligned source only if it
                                                  #        starts with a lowercase. Otherwise, copies the source word.
    ALIGN_FROM_RAW = True                         # Align using the full vocabulary or the short_list

    MAPPING = DATA_ROOT_PATH + '/mapping.%s_%s.pkl' % (SRC_LAN, TRG_LAN) # Source -- Target pkl mapping (used for heuristics 1--2)

    # Word representation params
    TOKENIZATION_METHOD = 'tokenize_none'         # Select which tokenization we'll apply.
                                                  # See Dataset class (from stager_keras_wrapper) for more info.
    DETOKENIZATION_METHOD = 'detokenize_bpe'       # Select which de-tokenization method we'll apply

    APPLY_DETOKENIZATION = False                  # Wheter we apply a detokenization method

    TOKENIZE_HYPOTHESES = True   		          # Whether we tokenize the hypotheses using the previously defined tokenization method
    TOKENIZE_REFERENCES = True                    # Whether we tokenize the references using the previously defined tokenization method
    # Input image parameters
    DATA_AUGMENTATION = False                     # Apply data augmentation on input data (still unimplemented for text inputs)

    # Text parameters
    FILL = 'end'                                  # Whether we pad the 'end' or the 'start' of the sentence with 0s
    PAD_ON_BATCH = True                           # Whether we take as many timesteps as the longest sequence of
                                                  # the batch or a fixed size (MAX_OUTPUT_TEXT_LEN)
    # Input text parameters
    INPUT_VOCABULARY_SIZE = 0                     # Size of the input vocabulary. Set to 0 for using all,
                                                  # otherwise it will be truncated to these most frequent words.
    MIN_OCCURRENCES_INPUT_VOCAB = 0               # Minimum number of occurrences allowed for the words in the input vocabulary.
                                                  # Set to 0 for using them all.
    MAX_INPUT_TEXT_LEN = 50                       # Maximum length of the input sequence

    # Output text parameters
    OUTPUT_VOCABULARY_SIZE = 0                    # Size of the input vocabulary. Set to 0 for using all,
                                                  # otherwise it will be truncated to these most frequent words.
    MIN_OCCURRENCES_OUTPUT_VOCAB = 0              # Minimum number of occurrences allowed for the words in the output vocabulary.
    MAX_OUTPUT_TEXT_LEN = 50                      # Maximum length of the output sequence
                                                  # set to 0 if we want to use the whole answer as a single class
    MAX_OUTPUT_TEXT_LEN_TEST = MAX_OUTPUT_TEXT_LEN * 3  # Maximum length of the output sequence during test time

    # Optimizer parameters (see model.compile() function)
    LOSS = 'categorical_crossentropy'
    CLASSIFIER_ACTIVATION = 'softmax'

    OPTIMIZER = 'Adam'                            # Optimizer
    LR = 0.001                                    # Learning rate. Recommended values - Adam 0.001 - Adadelta 1.0
    CLIP_C = 1.                                   # During training, clip L2 norm of gradients to this value (0. means deactivated)
    CLIP_V = 0.                                   # During training, clip absolute value of gradients to this value (0. means deactivated)
    SAMPLE_WEIGHTS = True                         # Select whether we use a weights matrix (mask) for the data outputs
    LR_DECAY = None                               # Minimum number of epochs before the next LR decay. Set to None if don't want to decay the learning rate
    LR_GAMMA = 0.8                                # Multiplier used for decreasing the LR

    # Training parameters
    MAX_EPOCH = 500                               # Stop when computed this number of epochs
    BATCH_SIZE = 50                               # Size of each minibatch

    HOMOGENEOUS_BATCHES = False                   # Use batches with homogeneous output lengths (Dangerous!!)
    JOINT_BATCHES = 4                             # When using homogeneous batches, get this number of batches to sort
    PARALLEL_LOADERS = 1                          # Parallel data batch loaders
    EPOCHS_FOR_SAVE = 1                           # Number of epochs between model saves
    WRITE_VALID_SAMPLES = True                    # Write valid samples in file
    SAVE_EACH_EVALUATION = True                   # Save each time we evaluate the model

    # Early stop parameters
    EARLY_STOP = True                             # Turns on/off the early stop protocol
    PATIENCE = 20                                 # We'll stop if the val STOP_METRIC does not improve after this
                                                  # number of evaluations
    STOP_METRIC = 'Bleu_4'                        # Metric for the stop

    # Model parameters
    MODEL_TYPE = 'GroundHogModel'                 # Model to train. See model_zoo() for the supported architectures
    RNN_TYPE = 'LSTM'                             # RNN unit type ('LSTM' and 'GRU' supported)
    INIT_FUNCTION = 'glorot_uniform'              # Initialization function for matrices (see keras/initializations.py)

    SOURCE_TEXT_EMBEDDING_SIZE = 64              # Source language word embedding size.
    SRC_PRETRAINED_VECTORS = None                 # Path to pretrained vectors (e.g.: DATA_ROOT_PATH + '/DATA/word2vec.%s.npy' % SRC_LAN)
                                                  # Set to None if you don't want to use pretrained vectors.
                                                  # When using pretrained word embeddings. this parameter must match with the word embeddings size
    SRC_PRETRAINED_VECTORS_TRAINABLE = True       # Finetune or not the target word embedding vectors.

    TARGET_TEXT_EMBEDDING_SIZE = 64              # Source language word embedding size.
    TRG_PRETRAINED_VECTORS = None                 # Path to pretrained vectors. (e.g. DATA_ROOT_PATH + '/DATA/word2vec.%s.npy' % TRG_LAN)
                                                  # Set to None if you don't want to use pretrained vectors.
                                                  # When using pretrained word embeddings, the size of the pretrained word embeddings must match with the word embeddings size.
    TRG_PRETRAINED_VECTORS_TRAINABLE = True       # Finetune or not the target word embedding vectors.

    # Encoder configuration
    ENCODER_HIDDEN_SIZE = 64                     # For models with RNN encoder
    BIDIRECTIONAL_ENCODER = True                  # Use bidirectional encoder
    N_LAYERS_ENCODER = 1                          # Stack this number of encoding layers
    BIDIRECTIONAL_DEEP_ENCODER = True             # Use bidirectional encoder in all encoding layers

    # Decoder configuration
    DECODER_HIDDEN_SIZE = 64                     # For models with RNN decoder
    N_LAYERS_DECODER = 1                          # Stack this number of decoding layers.
    ADDITIONAL_OUTPUT_MERGE_MODE = 'sum'          # Merge mode for the skip-connections
    ATTENTION_SIZE = DECODER_HIDDEN_SIZE 
    # Skip connections size
    SKIP_VECTORS_HIDDEN_SIZE = TARGET_TEXT_EMBEDDING_SIZE

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

    USE_DROPOUT = False                           # Use dropout
    DROPOUT_P = 0.5                               # Percentage of units to drop

    USE_RECURRENT_INPUT_DROPOUT = False           # Use dropout in input cells of recurrent layers
    RECURRENT_INPUT_DROPOUT_P = 0.5               # Percentage of units to drop in input cells of recurrent layers

    USE_RECURRENT_DROPOUT = False                 # Use dropout in recurrent layers # DANGEROUS!
    RECURRENT_DROPOUT_P = 0.5                     # Percentage of units to drop in recurrent layers

    USE_NOISE = True                              # Use gaussian noise during training
    NOISE_AMOUNT = 0.01                           # Amount of noise

    USE_BATCH_NORMALIZATION = True                # If True it is recommended to deactivate Dropout
    BATCH_NORMALIZATION_MODE = 1                  # See documentation in Keras' BN

    USE_PRELU = False                             # use PReLU activations as regularizer
    USE_L2 = False                                # L2 normalization on the features

    # Results plot and models storing parameters
    EXTRA_NAME = ''                               # This will be appended to the end of the model name
    MODEL_NAME = TASK_NAME + '_' + SRC_LAN + TRG_LAN + '_' + MODEL_TYPE + \
                 '_src_emb_' + str(SOURCE_TEXT_EMBEDDING_SIZE) + \
                 '_bidir_' + str(BIDIRECTIONAL_ENCODER) + \
                 '_enc_' + RNN_TYPE + '_' + str(ENCODER_HIDDEN_SIZE) + \
                 '_dec_' + RNN_TYPE + '_' + str(DECODER_HIDDEN_SIZE) + \
                 '_deepout_' + '_'.join([layer[0] for layer in DEEP_OUTPUT_LAYERS]) + \
                 '_trg_emb_' + str(TARGET_TEXT_EMBEDDING_SIZE) + \
                 '_' + OPTIMIZER + '_' + str(LR)

    MODEL_NAME += EXTRA_NAME

    STORE_PATH = 'trained_models/' + MODEL_NAME + '/'  # Models and evaluation results will be stored here
    DATASET_STORE_PATH = STORE_PATH + '/'              # Dataset instance will be stored here

    SAMPLING_SAVE_MODE = 'list'                        # 'list' or 'vqa'
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
