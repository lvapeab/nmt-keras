
def load_parameters():
    """
    Loads the defined hyperparameters
    :return parameters: Dictionary of loaded parameters
    """
    # Optimizer parameters (see model.compile() function)
    LOSS = 'categorical_crossentropy'
    CLASSIFIER_ACTIVATION = 'softmax'

    OPTIMIZER = 'PAS'                             # Optimizer
    LR = 0.01                                     # Learning rate.
    # PAS-like params
    C = 0.01                                      # Constant parameter for PAS and PPAS optimizer.
    N_BEST_OPTIMIZER = False                      # Use N-Best list-based optimization

    # General params
    CLIP_C = 0.                                   # During training, clip gradients to this norm
    LR_DECAY = None                               # Minimum number of epochs before the next LR decay. Set to None if don't want to decay the learning rate
    LR_GAMMA = 0.8                                # Multiplier used for decreasing the LR

    # Training parameters
    MAX_EPOCH = 1                                 # Stop when computed this number of epochs
    BATCH_SIZE = 1                                # Size of each minibatch

    EPOCHS_FOR_SAVE = 1                           # Number of epochs between model saves
    WRITE_VALID_SAMPLES = True                    # Write valid samples in file
    SAVE_EACH_EVALUATION = True                   # Save each time we evaluate the model

    # Early stop parameters
    EARLY_STOP = False                            # Turns on/off the early stop protocol

    # Model parameters
    MODEL_TYPE = 'GroundHogModel'                 # Model to train. See model_zoo() for the supported architectures
    NOISE_AMOUNT = 0.0                            # Amount of noise

    STORE_PATH = 'trained_models/retrained_model/'  # Models and evaluation results will be stored here

    # ================================================ #
    parameters = locals().copy()
    return parameters
