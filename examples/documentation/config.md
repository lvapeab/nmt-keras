# NMT-Keras' config file

 This document describes the available hyperparameters used for training NMT-Keras.


 ##### Input data params

  * **DATASET_NAME**: Task name. Used for naming and for indexing files.
  * **SRC_LAN**: Language of the source text. Used for naming.
  * **TRG_LAN**: Language of the target text. Used for naming and for coputing language-dependent metrics (e.g. Meteor)
  * **DATA_ROOT_PATH**: Path to the data
  * **TEXT_FILES**: Dictionary containing the splits ('train/val/test) and the files corresponding to each one. The source/target languages will be appended to these files.

 ##### Input/output params

  * **INPUTS_IDS_DATASET**: Name of the inputs of the Dataset class.
  * **OUTPUTS_IDS_DATASET**: Name of the inputs of the Dataset class.
  * **INPUTS_IDS_MODEL**:Name of the inputs of the Model.
  * **OUTPUTS_IDS_MODEL**: Name of the inputs of the Model.

  ##### Evaluation params
  * **METRICS**: List of metric used for evaluating the model. The `coco` package is recommended.
  * **EVAL_ON_SETS**: List of splits ('train', 'val', 'test') to evaluate with the metrics from METRICS. Typically: 'val'
  * **EVAL_ON_SETS_KERAS**: List of splits ('train', 'val', 'test') to evaluate with the Keras metrics.
  * **START_EVAL_ON_EPOCH**: The evaluation starts at this epoch.
  * **EVAL_EACH_EPOCHS**: Whether the evaluation frequency units are epochs or updates.
  * **EVAL_EACH**: Evaluation frequency.

  ##### Decoding parameters
  * **SAMPLING**: Decoding mode. Only 'max_likelihood' tested.
  * **TEMPERATURE**: Multinomial sampling temerature.
  * **BEAM_SEARCH**: Switches on-off the beam search.
  * **BEAM_SIZE**: Beam size.
  * **OPTIMIZED_SEARCH**: Encode the source only once per sample (recommended).
  * **NORMALIZE_SAMPLING**: Normalize hypotheses scores according to the length
  * **ALPHA_FACTOR**: Normalization according to length^ALPHA_FACTOR ([source](arxiv.org/abs/1609.08144))

  ##### Sampling params: Show some samples during training
  * **SAMPLE_ON_SETS**: Splits from where we'll sample.
  * **N_SAMPLES**: Number of samples generated
  * **START_SAMPLING_ON_EPOCH**: First epoch where to start the sampling counter
  * **SAMPLE_EACH_UPDATES**: Sampling frequency (always in #updates)

   ##### Unknown words treatment
   * POS_UNK: Enable unknown words replacement strategy.
   * HEURISTIC: Heuristic followed for replacing unks:
   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 0: Replace the UNK by the correspondingly aligned source.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 1: Replace the UNK by the translation (given by an [external dictionary](https://github.com/lvapeab/nmt-keras/blob/master/utils/build_mapping_file.sh)) of the aligned source.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  2: Replace the UNK by the translation (given by an  [external dictionary](https://github.com/lvapeab/nmt-keras/blob/master/utils/build_mapping_file.sh)) of the aligned source &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; only if it starts with a lowercase. Otherwise, copies the source word.
   * ALIGN_FROM_RAW: Align using the source files or the short-list model vocabulary.

   * MAPPING: Mapping dictionary path (for heuristics 1 and 2). Obtained with the [build_mapping_file](https://github.com/lvapeab/nmt-keras/blob/master/utils/build_mapping_file.sh) script.

   #####  Word representation params
   * TOKENIZATION_METHOD: Tokenization applied to the input and output text. 
   * DETOKENIZATION_METHOD: Detokenization applied to the input and output text. 
   * APPLY_DETOKENIZATION: Wheter we apply the detokenization method
   * TOKENIZE_HYPOTHESES: Whether we tokenize the hypotheses (for computing metrics).
   * TOKENIZE_REFERENCES = Whether we tokenize the references (for computing metrics).

    ##### Text parameters
    * FILL: Padding mode: Insert zeroes at the 'start', 'center' or 'end'.
    * PAD_ON_BATCH: Make batches of a fixed number of timesteps or pad to the maximum length of the minibatch.

    ##### Input text parameters
    INPUT_VOCABULARY_SIZE = 0                     # Size of the input vocabulary. Set to 0 for using all,
                                                  # otherwise it will be truncated to these most frequent words.
    MIN_OCCURRENCES_VOCAB = 0                     # Minimum number of occurrences allowed for the words in the vocabulay.
                                                  # Set to 0 for using them all.
    MAX_INPUT_TEXT_LEN = 50                       # Maximum length of the input sequence

    # Output text parameters
    OUTPUT_VOCABULARY_SIZE = 0                    # Size of the input vocabulary. Set to 0 for using all,
                                                  # otherwise it will be truncated to these most frequent words.
    MAX_OUTPUT_TEXT_LEN = 50                      # Maximum length of the output sequence
                                                  # set to 0 if we want to use the whole answer as a single class
    MAX_OUTPUT_TEXT_LEN_TEST = 120                # Maximum length of the output sequence during test time

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

    HOMOGENEOUS_BATCHES = False                   # Use batches with homogeneous output lengths
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

    SOURCE_TEXT_EMBEDDING_SIZE = 420              # Source language word embedding size.
    SRC_PRETRAINED_VECTORS = None                 # Path to pretrained vectors (e.g.: DATA_ROOT_PATH + '/DATA/word2vec.%s.npy' % SRC_LAN)
                                                  # Set to None if you don't want to use pretrained vectors.
                                                  # When using pretrained word embeddings. this parameter must match with the word embeddings size
    SRC_PRETRAINED_VECTORS_TRAINABLE = True       # Finetune or not the target word embedding vectors.

    TARGET_TEXT_EMBEDDING_SIZE = 420              # Source language word embedding size.
    TRG_PRETRAINED_VECTORS = None                 # Path to pretrained vectors. (e.g. DATA_ROOT_PATH + '/DATA/word2vec.%s.npy' % TRG_LAN)
                                                  # Set to None if you don't want to use pretrained vectors.
                                                  # When using pretrained word embeddings, the size of the pretrained word embeddings must match with the word embeddings size.
    TRG_PRETRAINED_VECTORS_TRAINABLE = True       # Finetune or not the target word embedding vectors.

    # Encoder configuration
    ENCODER_HIDDEN_SIZE = 600                     # For models with RNN encoder
    BIDIRECTIONAL_ENCODER = True                  # Use bidirectional encoder
    N_LAYERS_ENCODER = 1                          # Stack this number of encoding layers
    BIDIRECTIONAL_DEEP_ENCODER = True             # Use bidirectional encoder in all encoding layers

    # Decoder configuration
    DECODER_HIDDEN_SIZE = 600                     # For models with RNN decoder
    N_LAYERS_DECODER = 1                          # Stack this number of decoding layers (unimplemented)
    ADDITIONAL_OUTPUT_MERGE_MODE = 'sum'          # Merge mode for the skip-connections
    # Skip connections size
    SKIP_VECTORS_HIDDEN_SIZE = TARGET_TEXT_EMBEDDING_SIZE

    # Fully-Connected layers for initializing the first RNN state
    #       Here we should only specify the activation function of each layer
    #       (as they have a potentially fixed size)
    #       (e.g INIT_LAYERS = ['tanh', 'relu'])
    INIT_LAYERS = ['tanh']

    # Additional Fully-Connected layers's sizes applied before softmax.
    #       Here we should specify the activation function and the output dimension
    #       (e.g DEEP_OUTPUT_LAYERS = [('tanh', 600), ('relu', 400), ('relu', 200)])
    DEEP_OUTPUT_LAYERS = [('linear', TARGET_TEXT_EMBEDDING_SIZE)]

    # Regularizers
    WEIGHT_DECAY = 1e-4                           # L2 regularization
    RECURRENT_WEIGHT_DECAY = 0.                   # L2 regularization in recurrent layers

    USE_DROPOUT = False                           # Use dropout
    DROPOUT_P = 0.5                               # Percentage of units to drop

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
    MODEL_NAME = DATASET_NAME + '_' + SRC_LAN + TRG_LAN + '_' + MODEL_TYPE + \
                 '_src_emb_' + str(SOURCE_TEXT_EMBEDDING_SIZE) + \
                 '_bidir_' + str(BIDIRECTIONAL_ENCODER) + \
                 '_enc_' + RNN_TYPE + '_' + str(ENCODER_HIDDEN_SIZE) + \
                 '_dec_' + RNN_TYPE + '_' + str(DECODER_HIDDEN_SIZE) + \
                 '_deepout_' + '_'.join([layer[0] for layer in DEEP_OUTPUT_LAYERS]) + \
                 '_trg_emb_' + str(TARGET_TEXT_EMBEDDING_SIZE) + \
                 '_' + OPTIMIZER + '_' + str(LR)

    MODEL_NAME += EXTRA_NAME

    STORE_PATH = 'trained_models/' + MODEL_NAME + '/'  # Models and evaluation results will be stored here
    DATASET_STORE_PATH = 'datasets/'                   # Dataset instance will be stored here

    SAMPLING_SAVE_MODE = 'list'                        # 'list' or 'vqa'
    VERBOSE = 1                                        # Verbosity level
    RELOAD = 0                                         # If 0 start training from scratch, otherwise the model
                                                       # Saved on epoch 'RELOAD' will be used
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
