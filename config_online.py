
def load_parameters():
    """
    Loads the defined hyperparameters
    :return parameters: Dictionary of loaded parameters
    """

    # Optimizer parameters (see model.compile() function)
    LOSS = 'categorical_crossentropy'
    KERAS_METRICS = ['categorical_crossentropy']                            # Metric used for evaluating the model

    CLASSIFIER_ACTIVATION = 'softmax'
    STORE_PATH = 'trained_models/retrained_model/'  # Models and evaluation results will be stored here

    OPTIMIZER = 'SGD'                             # Optimizer
    LR = 0.01                                     # Learning rate. Recommended values - Adam 0.001 - Adadelta 1.0
    CLIP_C = 10.                                  # During training, clip gradients to this norm
    SAMPLE_WEIGHTS = True                         # Select whether we use a weights matrix (mask) for the data outputs
    LR_DECAY = None                               # Minimum number of epochs before the next LR decay. Set to None if don't want to decay the learning rate
    LR_GAMMA = 0.8                                # Multiplier used for decreasing the LR

    # Training parameters
    MAX_EPOCH = 1                               # Stop when computed this number of epochs
    BATCH_SIZE = 1                               # Size of each minibatch

    HOMOGENEOUS_BATCHES = False                   # Use batches with homogeneous output lengths for every minibatch (Possibly buggy!)
    PARALLEL_LOADERS = 8                          # Parallel data batch loaders
    EPOCHS_FOR_SAVE = 1                           # Number of epochs between model saves
    WRITE_VALID_SAMPLES = True                    # Write valid samples in file
    SAVE_EACH_EVALUATION = True                   # Save each time we evaluate the model

    # Early stop parameters
    EARLY_STOP = False                             # Turns on/off the early stop protocol
    PATIENCE = 0                                 # We'll stop if the val STOP_METRIC does not improve after this
                                                  # number of evaluations
    STOP_METRIC = 'Bleu_4'                        # Metric for the stop

    # Model parameters
    MODEL_TYPE = 'GroundHogModel'                 # Model to train. See model_zoo() for the supported architectures
    RNN_TYPE = 'GRU'                              # RNN unit type ('LSTM' and 'GRU' supported)

    # Regularizers
    WEIGHT_DECAY = 0.                             # L2 regularization
    RECURRENT_WEIGHT_DECAY = 0.                   # L2 regularization in recurrent layers

    USE_DROPOUT = False                           # Use dropout
    DROPOUT_P = 0.5                               # Percentage of units to drop

    USE_RECURRENT_DROPOUT = False                 # Use dropout in recurrent layers # DANGEROUS!
    RECURRENT_DROPOUT_P = 0.5                     # Percentage of units to drop in recurrent layers

    USE_NOISE = False                              # Use gaussian noise during training
    NOISE_AMOUNT = 0.0                         # Amount of noise

    USE_BATCH_NORMALIZATION = True                # If True it is recommended to deactivate Dropout
    BATCH_NORMALIZATION_MODE = 1                  # See documentation in Keras' BN

    USE_PRELU = False                             # use PReLU activations as regularizer
    USE_L2 = False                                # L2 normalization on the features

    # ================================================ #
    parameters = locals().copy()
    return parameters
