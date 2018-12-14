
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
    K = 1                                         # Number of iterations to perform per sample
    USE_CUSTOM_LOSS = False if 'categorical_crossentropy' in LOSS else True
    N_BEST_OPTIMIZER = False                      # Use N-Best list-based optimization
    OPTIMIZER_REGULARIZER = ''                    # Metric to optimize (BLEU or TER)

    # Learning rate scheduling params
    LR_DECAY = None                               # Minimum number of epochs before the next LR decay. Set to None if don't want to decay the learning rate
    LR_GAMMA = 0.8                                # Multiplier used for decreasing the LR


    # Training parameters
    MAX_EPOCH = 1                                 # Stop when computed this number of epochs

    EPOCHS_FOR_SAVE = 1                           # Number of epochs between model saves
    WRITE_VALID_SAMPLES = True                    # Write valid samples in file
    SAVE_EACH_EVALUATION = True                   # Save each time we evaluate the model

    # Early stop parameters
    EARLY_STOP = False                            # Turns on/off the early stop protocol

    # Model parameters
    NOISE_AMOUNT = 0.0                            # Amount of noise
    TRAIN_ONLY_LAST_LAYER = False

    STORE_PATH = original_params.get('STORE_PATH', 'trained_models') + ' /retrained_model/'  # Models and evaluation results will be stored here

    # ================================================ #
    parameters = locals().copy()
    return parameters
