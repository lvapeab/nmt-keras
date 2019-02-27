
def load_parameters(original_params):
    """
    Loads the defined hyperparameters
    :return parameters: Dictionary of loaded parameters
    """
    # Optimizer parameters (see model.compile() function)
    CLASSIFIER_ACTIVATION = 'softmax'
    LOSS = 'categorical_crossentropy'
    OPTIMIZER = 'adadelta'                        # Optimizer

    # Optimizer hyperparameters
    LR = 0.1                                      # Learning rate.
    QUASI_HYPERBOLIC_MOMENTUM = 0.7               # Quasi-hyperbolic momentum factor
    DAMPENING = False                             # Momentum dampening
    CLIP_C = 0.                                   # During training, clip gradients to this norm

    # PAS-like params
    C = 0.1                                       # Constant parameter for PAS and PPAS optimizer.
    D = 0.1                                       # Constant parameter for PAS and PPAS optimizer.

    K = 1                                         # Maximum number of iterations to perform per sample
    LR_HYP = LR                                   # Learning rate for hypothesis gradient-ascent (for minmax_categorical_crossentropy loss)

    USE_CUSTOM_LOSS = False if 'categorical_crossentropy' in LOSS else True
    N_BEST_OPTIMIZER = False                      # Use N-Best list-based optimization
    OPTIMIZER_REGULARIZER = ''                    # Metric to optimize (BLEU or TER)
    # Learning rate scheduling params
    LR_DECAY = None                               # Minimum number of epochs before the next LR decay. Set to None if don't want to decay the learning rate
    LR_GAMMA = 0.8                                # Multiplier used for decreasing the LR

    LABEL_SMOOTHING = 0.

    # Training parameters
    MAX_EPOCH = 1                                 # Stop when computed this number of epochs

    EPOCHS_FOR_SAVE = 1                           # Number of epochs between model saves
    WRITE_VALID_SAMPLES = True                    # Write valid samples in file
    SAVE_EACH_EVALUATION = True                   # Save each time we evaluate the model

    # Early stop parameters
    EARLY_STOP = False                            # Turns on/off the early stop protocol

    REGULARIZATION_FN = ''                        # Regularization function. 'L1', 'L2' and 'L1_L2' supported.
    WEIGHT_DECAY = 0.                             # Regularization coefficient.
    RECURRENT_WEIGHT_DECAY = 0.                   # Regularization coefficient in recurrent layers.
    DOUBLE_STOCHASTIC_ATTENTION_REG = 0.0         # Doubly stochastic attention (Eq. 14 from arXiv:1502.03044).

    # DROPOUT_P = 0.                              # Percentage of units to drop (0 means no dropout).
    # RECURRENT_INPUT_DROPOUT_P = 0.              # Percentage of units to drop in input cells of recurrent layers.
    # RECURRENT_DROPOUT_P = 0.                    # Percentage of units to drop in recurrent layers.
    # ATTENTION_DROPOUT_P = 0.                    # Percentage of units to drop in attention layers (0 means no dropout).

    USE_NOISE = False                             # Use gaussian noise during training.
    NOISE_AMOUNT = 0.0                            # Amount of noise.


    STORE_PATH = original_params.get('STORE_PATH', 'trained_models') + ' /retrained_model/'  # Models and evaluation results will be stored here

    # ================================================ #
    parameters = locals().copy()
    return parameters
