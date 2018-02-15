
def load_parameters():
    """
    Loads the defined hyperparameters
    :return parameters: Dictionary of loaded parameters
    """
    # Optimizer parameters (see model.compile() function)
    CLASSIFIER_ACTIVATION = 'softmax'
    LOSS = 'categorical_crossentropy'
    OPTIMIZER = 'adadelta'                        # Optimizer
    LR = 0.1                                      # Learning rate.
    # PAS-like params
    C = 0.1                                       # Constant parameter for PAS and PPAS optimizer.
    D = 0.1                                       # Constant parameter for PAS and PPAS optimizer.
    K = 1                                         # Number of iterations to perform per sample
    USE_CUSTOM_LOSS = False if 'categorical_crossentropy' in LOSS else True
    N_BEST_OPTIMIZER = False                      # Use N-Best list-based optimization
    OPTIMIZER_REGULARIZER = ''                    # Metric to optimize (BLEU or TER)

    # General params
    CLIP_C = 5.                                   # During training, clip gradients to this norm
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
    MODEL_TYPE = 'AttentionRNNEncoderDecoder'     # Model to train. See model_zoo() for the supported architectures
    NOISE_AMOUNT = 0.0                            # Amount of noise

    STORE_PATH = 'trained_models/retrained_model2/'  # Models and evaluation results will be stored here

    # ================================================ #
    parameters = locals().copy()
    return parameters
