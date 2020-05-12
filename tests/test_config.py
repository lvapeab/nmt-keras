import tempfile
import shutil


def load_tests_params():
    """
    Loads the hyperparameters for tests.
    :return parameters: Dictionary of loaded parameters.
    """

    # Input data params
    TASK_NAME = 'EuTrans'  # Task name.
    DATASET_NAME = TASK_NAME  # Dataset name.
    SRC_LAN = 'es'  # Language of the source text.
    TRG_LAN = 'en'  # Language of the target text.
    DATA_ROOT_PATH = 'examples/%s/' % DATASET_NAME  # Path where data is stored.

    # SRC_LAN or TRG_LAN will be added to the file names.
    TEXT_FILES = {'train': 'dev.',  # Data files.
                  'val': 'dev.',
                  'test': 'dev.'}

    GLOSSARY = None  # Glossary location. If not None, it overwrites translations according to this glossary file

    # Dataset class parameters
    INPUTS_IDS_DATASET = ['source_text', 'state_below']  # Corresponding inputs of the dataset.
    OUTPUTS_IDS_DATASET = ['target_text']  # Corresponding outputs of the dataset.
    INPUTS_IDS_MODEL = ['source_text', 'state_below']  # Corresponding inputs of the built model.
    OUTPUTS_IDS_MODEL = ['target_text']  # Corresponding outputs of the built model.
    INPUTS_TYPES_DATASET = ['text-features',
                            'text-features']  # Corresponding types of the data. 'text' or 'text-features' allowed.
    OUTPUTS_TYPES_DATASET = ['text-features']  # They are equivalent, only differ on how the data is loaded.

    # Evaluation params
    METRICS = ['sacrebleu', 'perplexity']  # Metric used for evaluating the model.
    KERAS_METRICS = ['perplexity']  # Metrics to be logged by Keras during training (in addition to the loss).
    EVAL_ON_SETS = ['val']  # Possible values: 'train', 'val' and 'test' (external evaluator).
    START_EVAL_ON_EPOCH = 0  # First epoch to start the model evaluation.
    EVAL_EACH_EPOCHS = True  # Select whether evaluate between N epochs or N updates.
    EVAL_EACH = 1  # Sets the evaluation frequency (epochs or updates).

    # Search parameters
    SAMPLING = 'max_likelihood'  # Possible values: multinomial or max_likelihood (recommended).
    TEMPERATURE = 1  # Multinomial sampling parameter.
    BEAM_SEARCH = True  # Switches on-off the beam search procedure.
    BEAM_SIZE = 6  # Beam size (in case of BEAM_SEARCH == True).
    OPTIMIZED_SEARCH = True  # Compute annotations only a single time per sample.
    SEARCH_PRUNING = False  # Apply pruning strategies to the beam search method.
    # It will likely increase decoding speed, but decrease quality.
    MAXLEN_GIVEN_X = False  # Generate translations of similar length to the source sentences.
    MAXLEN_GIVEN_X_FACTOR = 2  # The hypotheses will have (as maximum) the number of words of the
    # source sentence * LENGTH_Y_GIVEN_X_FACTOR.
    MINLEN_GIVEN_X = False  # Generate translations of similar length to the source sentences.
    MINLEN_GIVEN_X_FACTOR = 3  # The hypotheses will have (as minimum) the number of words of the
    # source sentence / LENGTH_Y_GIVEN_X_FACTOR.

    # Apply length and coverage decoding normalizations.
    # See Section 7 from Wu et al. (2016) (https://arxiv.org/abs/1609.08144).
    LENGTH_PENALTY = False  # Apply length penalty.
    LENGTH_NORM_FACTOR = 0.2  # Length penalty factor.
    COVERAGE_PENALTY = False  # Apply source coverage penalty.
    COVERAGE_NORM_FACTOR = 0.2  # Coverage penalty factor.

    # Alternative (simple) length normalization.
    NORMALIZE_SAMPLING = False  # Normalize hypotheses scores according to their length:
    ALPHA_FACTOR = .6  # Normalization according to |h|**ALPHA_FACTOR.

    # Sampling params: Show some samples during training.
    SAMPLE_ON_SETS = ['train', 'val']  # Possible values: 'train', 'val' and 'test'.
    N_SAMPLES = 5  # Number of samples generated.
    START_SAMPLING_ON_EPOCH = 1  # First epoch where to start the sampling counter.
    SAMPLE_EACH_UPDATES = 300  # Sampling frequency (always in #updates).

    # Unknown words treatment
    POS_UNK = False  # Enable POS_UNK strategy for unknown words.
    HEURISTIC = 0  # Heuristic to follow:
    #     0: Replace the UNK by the correspondingly aligned source.
    #     1: Replace the UNK by the translation (given by an external
    #        dictionary) of the correspondingly aligned source.
    #     2: Replace the UNK by the translation (given by an external
    #        dictionary) of the correspondingly aligned source only if it
    #        starts with a lowercase. Otherwise, copies the source word.
    ALIGN_FROM_RAW = True  # Align using the full vocabulary or the short_list.

    # Source -- Target pkl mapping (used for heuristics 1--2). See utils/build_mapping_file.sh for further info.
    MAPPING = DATA_ROOT_PATH + '/mapping.%s_%s.pkl' % (SRC_LAN, TRG_LAN)

    # Word representation params
    TOKENIZATION_METHOD = 'tokenize_none'  # Select which tokenization we'll apply.
    # See Dataset class (from stager_keras_wrapper) for more info.
    BPE_CODES_PATH = DATA_ROOT_PATH + '/training_codes.joint'  # If TOKENIZATION_METHOD = 'tokenize_bpe',
    # sets the path to the learned BPE codes.
    DETOKENIZATION_METHOD = 'detokenize_none'  # Select which de-tokenization method we'll apply.

    APPLY_DETOKENIZATION = False  # Wheter we apply a detokenization method.

    TOKENIZE_HYPOTHESES = True  # Whether we tokenize the hypotheses using the
    # previously defined tokenization method.
    TOKENIZE_REFERENCES = True  # Whether we tokenize the references using the
    # previously defined tokenization method.

    # Input image parameters
    DATA_AUGMENTATION = False  # Apply data augmentation on input data (still unimplemented for text inputs).

    # Text parameters
    FILL = 'end'  # Whether we pad the 'end', the 'start' of  the sentence with 0s. We can also 'center' it.
    PAD_ON_BATCH = True  # Whether we take as many timesteps as the longest sequence of
    # the batch or a fixed size (MAX_OUTPUT_TEXT_LEN).
    # Input text parameters
    INPUT_VOCABULARY_SIZE = 0  # Size of the input vocabulary. Set to 0 for using all,
    # otherwise it will be truncated to these most frequent words.
    MIN_OCCURRENCES_INPUT_VOCAB = 0  # Minimum number of occurrences allowed for the words in the input vocabulary.
    # Set to 0 for using them all.
    MAX_INPUT_TEXT_LEN = 50  # Maximum length of the input sequence.

    # Output text parameters
    OUTPUT_VOCABULARY_SIZE = 0  # Size of the input vocabulary. Set to 0 for using all,
    # otherwise it will be truncated to these most frequent words.
    MIN_OCCURRENCES_OUTPUT_VOCAB = 0  # Minimum number of occurrences allowed for the words in the output vocabulary.
    MAX_OUTPUT_TEXT_LEN = 10  # Maximum length of the output sequence.
    # set to 0 if we want to use the whole answer as a single class.
    MAX_OUTPUT_TEXT_LEN_TEST = MAX_OUTPUT_TEXT_LEN  # Maximum length of the output sequence during test time.

    # Optimizer parameters (see model.compile() function).
    LOSS = 'categorical_crossentropy'
    CLASSIFIER_ACTIVATION = 'softmax'
    SAMPLE_WEIGHTS = True  # Select whether we use a weights matrix (mask) for the data outputs
    LABEL_SMOOTHING = 0.  # Epsilon value for label smoothing. Only valid for 'categorical_crossentropy' loss. See arxiv.org/abs/1512.00567.

    OPTIMIZER = 'Adam'  # Optimizer. Supported optimizers: SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam.
    LR = 0.001  # Learning rate. Recommended values - Adam 0.0002 - Adadelta 1.0.
    CLIP_C = 5.  # During training, clip L2 norm of gradients to this value (0. means deactivated).
    CLIP_V = 0.  # During training, clip absolute value of gradients to this value (0. means deactivated).
    USE_TF_OPTIMIZER = True  # Use native Tensorflow's optimizer (only for the Tensorflow backend).

    # Advanced parameters for optimizers. Default values are usually effective.
    MOMENTUM = 0.  # Momentum value (for SGD optimizer).
    NESTEROV_MOMENTUM = False  # Use Nesterov momentum (for SGD optimizer).
    RHO = 0.9  # Rho value (for Adadelta and RMSprop optimizers).
    BETA_1 = 0.9  # Beta 1 value (for Adam, Adamax Nadam optimizers).
    BETA_2 = 0.999  # Beta 2 value (for Adam, Adamax Nadam optimizers).
    AMSGRAD = False  # Whether to apply the AMSGrad variant of Adam (see https://openreview.net/pdf?id=ryQu7f-RZ).
    EPSILON = 1e-8  # Optimizers epsilon value.
    ACCUMULATE_GRADIENTS = 1  # Accumulate gradients for this number of batches. Currently only implemented for Adam.

    # Learning rate schedule
    LR_DECAY = None  # Frequency (number of epochs or updates) between LR annealings. Set to None for not decay the learning rate.
    LR_GAMMA = 0.8  # Multiplier used for decreasing the LR.
    LR_REDUCE_EACH_EPOCHS = False  # Reduce each LR_DECAY number of epochs or updates.
    LR_START_REDUCTION_ON_EPOCH = 0  # Epoch to start the reduction.
    LR_REDUCER_TYPE = 'exponential'  # Function to reduce. 'linear' and 'exponential' implemented.
    # Linear reduction: new_lr = lr * LR_GAMMA
    # Exponential reduction: new_lr = lr * LR_REDUCER_EXP_BASE ** (current_nb / LR_HALF_LIFE) * LR_GAMMA
    # Noam reduction: new_lr = lr * min(current_nb ** LR_REDUCER_EXP_BASE, current_nb * LR_HALF_LIFE ** WARMUP_EXP)
    LR_REDUCER_EXP_BASE = -0.5  # Base for the exponential decay.
    LR_HALF_LIFE = 100  # Factor/warmup steps for exponenital/noam decay.
    WARMUP_EXP = -1.5  # Warmup steps for noam decay.
    MIN_LR = 1e-9  # Minimum value allowed for the decayed LR

    # Training parameters
    MAX_EPOCH = 1  # Stop when computed this number of epochs.
    BATCH_SIZE = 10  # Size of each minibatch.
    N_GPUS = 1  # Number of GPUs to use. Only for Tensorflow backend. Each GPU will receive mini-batches of BATCH_SIZE / N_GPUS.

    HOMOGENEOUS_BATCHES = False  # Use batches with homogeneous output lengths (Dangerous!!).
    JOINT_BATCHES = 4  # When using homogeneous batches, get this number of batches to sort.
    PARALLEL_LOADERS = 1  # Parallel data batch loaders. Somewhat untested if > 1.
    EPOCHS_FOR_SAVE = 1  # Number of epochs between model saves.
    WRITE_VALID_SAMPLES = True  # Write valid samples in file.
    SAVE_EACH_EVALUATION = True  # Save each time we evaluate the model.

    # Early stop parameters
    EARLY_STOP = True  # Turns on/off the early stop protocol.
    PATIENCE = 10  # We'll stop if the val STOP_METRIC does not improve after this.
    # number of evaluations.
    STOP_METRIC = 'Bleu_4'  # Metric for the stop.
    MIN_DELTA = 0.  # Minimum change in the monitored quantity to consider it as an improvement.

    # Model parameters
    MODEL_TYPE = 'AttentionRNNEncoderDecoder'  # Model to train. See model_zoo.py for more info.
    # Supported architectures: 'AttentionRNNEncoderDecoder' and 'Transformer'.

    # Common hyperparameters for all models
    # # # # # # # # # # # # # # # # # # # # # # # #
    TRAINABLE_ENCODER = True  # Whether the encoder's weights should be modified during training.
    TRAINABLE_DECODER = True  # Whether the decoder's weights should be modified during training.

    # Initializers (see keras/initializations.py).
    INIT_FUNCTION = 'glorot_uniform'  # General initialization function for matrices.
    INNER_INIT = 'orthogonal'  # Initialization function for inner RNN matrices.
    INIT_ATT = 'glorot_uniform'  # Initialization function for attention mechism matrices

    SOURCE_TEXT_EMBEDDING_SIZE = 8  # Source language word embedding size.
    SRC_PRETRAINED_VECTORS = None  # Path to pretrained vectors (e.g.: DATA_ROOT_PATH + '/DATA/word2vec.%s.npy' % SRC_LAN).
    # Set to None if you don't want to use pretrained vectors.
    # When using pretrained word embeddings. this parameter must match with the word embeddings size
    SRC_PRETRAINED_VECTORS_TRAINABLE = True  # Finetune or not the target word embedding vectors.

    TARGET_TEXT_EMBEDDING_SIZE = 8  # Source language word embedding size.
    TRG_PRETRAINED_VECTORS = None  # Path to pretrained vectors. (e.g. DATA_ROOT_PATH + '/DATA/word2vec.%s.npy' % TRG_LAN)
    # Set to None if you don't want to use pretrained vectors.
    # When using pretrained word embeddings, the size of the pretrained word embeddings must match with the word embeddings size.
    TRG_PRETRAINED_VECTORS_TRAINABLE = True  # Finetune or not the target word embedding vectors.

    SCALE_SOURCE_WORD_EMBEDDINGS = False  # Scale source word embeddings by Sqrt(SOURCE_TEXT_EMBEDDING_SIZE).
    SCALE_TARGET_WORD_EMBEDDINGS = False  # Scale target word embeddings by Sqrt(TARGET_TEXT_EMBEDDING_SIZE).

    TIE_EMBEDDINGS = False  # Use the same embeddings for source and target language.

    N_LAYERS_ENCODER = 1  # Stack this number of encoding layers.
    N_LAYERS_DECODER = 1  # Stack this number of decoding layers.

    # Additional Fully-Connected layers applied before softmax.
    #       Here we should specify the activation function and the output dimension.
    #       (e.g DEEP_OUTPUT_LAYERS = [('tanh', 600), ('relu', 400), ('relu', 200)])
    DEEP_OUTPUT_LAYERS = [('linear', TARGET_TEXT_EMBEDDING_SIZE)]
    # # # # # # # # # # # # # # # # # # # # # # # #

    # AttentionRNNEncoderDecoder model hyperparameters
    # # # # # # # # # # # # # # # # # # # # # # # #
    ENCODER_RNN_TYPE = 'LSTM'  # Encoder's RNN unit type ('LSTM' and 'GRU' supported).
    USE_CUDNN = False  # Use CuDNN's implementation of GRU and LSTM (only for Tensorflow backend).

    DECODER_RNN_TYPE = 'ConditionalLSTM'  # Decoder's RNN unit type.
    # ('LSTM', 'GRU', 'ConditionalLSTM' and 'ConditionalGRU' supported).
    ATTENTION_MODE = 'add'  # Attention mode. 'add' (Bahdanau-style), 'dot' (Luong-style) or 'scaled-dot'.

    # Encoder configuration
    ENCODER_HIDDEN_SIZE = 4  # For models with RNN encoder.
    BIDIRECTIONAL_ENCODER = True  # Use bidirectional encoder.
    BIDIRECTIONAL_DEEP_ENCODER = True  # Use bidirectional encoder in all encoding layers.
    BIDIRECTIONAL_MERGE_MODE = 'concat'  # Merge function for bidirectional layers.

    # Fully-Connected layers for initializing the first decoder RNN state.
    #       Here we should only specify the activation function of each layer (as they have a potentially fixed size)
    #       (e.g INIT_LAYERS = ['tanh', 'relu'])
    INIT_LAYERS = ['tanh']

    # Decoder configuration
    DECODER_HIDDEN_SIZE = 4  # For models with RNN decoder.
    ATTENTION_SIZE = DECODER_HIDDEN_SIZE

    # Skip connections parameters
    SKIP_VECTORS_HIDDEN_SIZE = TARGET_TEXT_EMBEDDING_SIZE  # Hidden size.
    ADDITIONAL_OUTPUT_MERGE_MODE = 'Add'  # Merge mode for the skip-connections (see keras.layers.merge.py).
    SKIP_VECTORS_SHARED_ACTIVATION = 'tanh'  # Activation for the skip vectors.
    # # # # # # # # # # # # # # # # # # # # # # # #

    # Transformer model hyperparameters
    # # # # # # # # # # # # # # # # # # # # # # # #
    MODEL_SIZE = 32  # Transformer model size (d_{model} in de paper).
    MULTIHEAD_ATTENTION_ACTIVATION = 'linear'  # Activation the input projections in the Multi-Head Attention blocks.
    FF_SIZE = MODEL_SIZE * 4  # Size of the feed-forward layers of the Transformer model.
    N_HEADS = 8  # Number of parallel attention layers of the Transformer model.
    # # # # # # # # # # # # # # # # # # # # # # # #

    # Regularizers
    REGULARIZATION_FN = 'L2'  # Regularization function. 'L1', 'L2' and 'L1_L2' supported.
    WEIGHT_DECAY = 1e-4  # Regularization coefficient.
    RECURRENT_WEIGHT_DECAY = 1e-4  # Regularization coefficient in recurrent layers.

    DROPOUT_P = 0.01  # Percentage of units to drop (0 means no dropout).
    RECURRENT_INPUT_DROPOUT_P = 0.01  # Percentage of units to drop in input cells of recurrent layers.
    RECURRENT_DROPOUT_P = 0.01  # Percentage of units to drop in recurrent layers.
    ATTENTION_DROPOUT_P = 0.  # Percentage of units to drop in attention layers (0 means no dropout).

    USE_NOISE = True  # Use gaussian noise during training.
    NOISE_AMOUNT = 0.01  # Amount of noise.

    USE_BATCH_NORMALIZATION = True  # If True it is recommended to deactivate Dropout.
    BATCH_NORMALIZATION_MODE = 1  # See documentation in Keras' BN.

    USE_PRELU = False  # use PReLU activations as regularizer.
    USE_L1 = False  # L1 normalization on the features.
    USE_L2 = False  # L2 normalization on the features.

    DOUBLE_STOCHASTIC_ATTENTION_REG = 0.7  # Doubly stochastic attention (Eq. 14 from arXiv:1502.03044).

    # Results plot and models storing parameters.
    EXTRA_NAME = ''  # This will be appended to the end of the model name.
    if MODEL_TYPE == 'AttentionRNNEncoderDecoder':
        MODEL_NAME = TASK_NAME + '_' + SRC_LAN + TRG_LAN + '_' + MODEL_TYPE + \
                     '_src_emb_' + str(SOURCE_TEXT_EMBEDDING_SIZE) + \
                     '_bidir_' + str(BIDIRECTIONAL_ENCODER) + \
                     '_enc_' + ENCODER_RNN_TYPE + '_' + str(ENCODER_HIDDEN_SIZE) + \
                     '_dec_' + DECODER_RNN_TYPE + '_' + str(DECODER_HIDDEN_SIZE) + \
                     '_deepout_' + '_'.join([layer[0] for layer in DEEP_OUTPUT_LAYERS]) + \
                     '_trg_emb_' + str(TARGET_TEXT_EMBEDDING_SIZE) + \
                     '_' + OPTIMIZER + '_' + str(LR)
    elif MODEL_TYPE == 'Transformer':
        MODEL_NAME = TASK_NAME + '_' + SRC_LAN + TRG_LAN + '_' + MODEL_TYPE + \
                     '_model_size_' + str(MODEL_SIZE) + \
                     '_ff_size_' + str(FF_SIZE) + \
                     '_num_heads_' + str(N_HEADS) + \
                     '_encoder_blocks_' + str(N_LAYERS_ENCODER) + \
                     '_decoder_blocks_' + str(N_LAYERS_DECODER) + \
                     '_deepout_' + '_'.join([layer[0] for layer in DEEP_OUTPUT_LAYERS]) + \
                     '_' + OPTIMIZER + '_' + str(LR)
    else:
        MODEL_NAME = TASK_NAME + '_' + SRC_LAN + TRG_LAN + '_' + \
                     MODEL_TYPE + '_' + OPTIMIZER + '_' + str(LR)

    MODEL_NAME += EXTRA_NAME

    STORE_PATH = tempfile.mkdtemp()  # Models and evaluation results will be stored here.
    DATASET_STORE_PATH = tempfile.mkdtemp()  # Dataset instance will be stored here.

    # Tensorboard configuration. Only if the backend is Tensorflow. Otherwise, it will be ignored.
    TENSORBOARD = True  # Switches On/Off the tensorboard callback.
    LOG_DIR = 'tensorboard_logs'  # Directory to store teh model. Will be created inside STORE_PATH.
    EMBEDDINGS_FREQ = 1  # Frequency (in epochs) at which selected embedding layers will be saved.

    SAMPLING_SAVE_MODE = 'list'  # 'list': Store in a text file, one sentence per line.
    PLOT_EVALUATION = False  # If True, the evaluation will be plotted into the model folder.

    VERBOSE = 1  # Verbosity level.
    RELOAD = 0  # If 0 start training from scratch, otherwise the model.
    # Saved on epoch 'RELOAD' will be used.
    RELOAD_EPOCH = True  # Select whether we reload epoch or update number.

    REBUILD_DATASET = True  # Build again or use stored instance.
    MODE = 'training'  # 'training' or 'sampling' (if 'sampling' then RELOAD must
    # be greater than 0 and EVAL_ON_SETS will be used).

    # Extra parameters for special trainings. In most cases, they should be set to `False`
    TRAIN_ON_TRAINVAL = False  # train the model on both training and validation sets combined.
    FORCE_RELOAD_VOCABULARY = False  # force building a new vocabulary from the training samples
    # applicable if RELOAD > 1
    # ================================================ #
    parameters = locals().copy()
    return parameters


def load_transformer_test_params():
    params = load_tests_params()
    params['MODEL_TYPE'] = 'Transformer'
    params['N_LAYERS_ENCODER'] = 2
    params['N_LAYERS_DECODER'] = 2
    params['MULTIHEAD_ATTENTION_ACTIVATION'] = 'relu'
    params['MODEL_SIZE'] = 8
    params['FF_SIZE'] = params['MODEL_SIZE'] * 4
    params['N_HEADS'] = 2
    params['REBUILD_DATASET'] = True
    params['OPTIMIZED_SEARCH'] = True
    params['POS_UNK'] = False
    return params


def clean_dirs(params):
    shutil.rmtree(params['STORE_PATH'])
    shutil.rmtree(params['DATASET_STORE_PATH'])
