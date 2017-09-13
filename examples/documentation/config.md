# NMT-Keras' config file

 This document describes the available hyperparameters used for training NMT-Keras.


 #### Input data params: Naming the task and setting the paths to the data files.
  * **DATASET_NAME**: Task name. Used for naming and for indexing files.
  * **SRC_LAN**: Language of the source text. Used for naming.
  * **TRG_LAN**: Language of the target text. Used for naming and for coputing language-dependent metrics (e.g. Meteor)
  * **DATA_ROOT_PATH**: Path to the data
  * **TEXT_FILES**: Dictionary containing the splits ('train/val/test) and the files corresponding to each one. The source/target languages will be appended to these files.

 #### Input/output params
    Parameters for naming the task and setting the paths to the data files.
  * **INPUTS_IDS_DATASET**: Name of the inputs of the Dataset class.
  * **OUTPUTS_IDS_DATASET**: Name of the outputs of the Dataset class.
  * **INPUTS_IDS_MODEL**: Name of the inputs of the Model.
  * **OUTPUTS_IDS_MODEL**: Name of the outputs of the Model.

  #### Evaluation params
  * **METRICS**: List of metric used for evaluating the model. The `coco` package is recommended.
  * **EVAL_ON_SETS**: List of splits ('train', 'val', 'test') to evaluate with the metrics from METRICS. Typically: 'val'
  * **EVAL_ON_SETS_KERAS**: List of splits ('train', 'val', 'test') to evaluate with the Keras metrics.
  * **START_EVAL_ON_EPOCH**: The evaluation starts at this epoch.
  * **EVAL_EACH_EPOCHS**: Whether the evaluation frequency units are epochs or updates.
  * **EVAL_EACH**: Evaluation frequency.

  #### Decoding parameters
  * **SAMPLING**: Decoding mode. Only 'max_likelihood' tested.
  * **TEMPERATURE**: Multinomial sampling temerature.
  * **BEAM_SEARCH**: Switches on-off the beam search.
  * **BEAM_SIZE**: Beam size.
  * **OPTIMIZED_SEARCH**: Encode the source only once per sample (recommended).
  * **NORMALIZE_SAMPLING**: Normalize hypotheses scores according to the length
  * **ALPHA_FACTOR**: Normalization according to length^ALPHA_FACTOR ([source](arxiv.org/abs/1609.08144))

  #### Sampling params: Show some samples during training
  * **SAMPLE_ON_SETS**: Splits from where we'll sample.
  * **N_SAMPLES**: Number of samples generated
  * **START_SAMPLING_ON_EPOCH**: First epoch where to start the sampling counter
  * **SAMPLE_EACH_UPDATES**: Sampling frequency (always in #updates)

   #### Unknown words treatment
   * **POS_UNK**: Enable unknown words replacement strategy.
   * **HEURISTIC**: Heuristic followed for replacing unks:
   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 0: Replace the UNK by the correspondingly aligned source.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 1: Replace the UNK by the translation (given by an [external dictionary](https://github.com/lvapeab/nmt-keras/blob/master/utils/build_mapping_file.sh)) of the aligned source.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  2: Replace the UNK by the translation (given by an  [external dictionary](https://github.com/lvapeab/nmt-keras/blob/master/utils/build_mapping_file.sh)) of the aligned source &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; only if it starts with a lowercase. Otherwise, copies the source word.
   * **ALIGN_FROM_RAW**: Align using the source files or the short-list model vocabulary.

   * **MAPPING**: Mapping dictionary path (for heuristics 1 and 2). Obtained with the [build_mapping_file](https://github.com/lvapeab/nmt-keras/blob/master/utils/build_mapping_file.sh) script.

   ####  Word representation params
   * **TOKENIZATION_METHOD**: Tokenization applied to the input and output text. 
   * **DETOKENIZATION_METHOD**: Detokenization applied to the input and output text. 
   * **APPLY_DETOKENIZATION**: Wheter we apply the detokenization method
   * **TOKENIZE_HYPOTHESES**: Whether we tokenize the hypotheses (for computing metrics).
   * **TOKENIZE_REFERENCES**: Whether we tokenize the references (for computing metrics).

   #### Text parameters
   * **FILL**: Padding mode: Insert zeroes at the 'start', 'center' or 'end'.
   * **PAD_ON_BATCH**: Make batches of a fixed number of timesteps or pad to the maximum length of the minibatch.

   #### Input text parameters
   * **INPUT_VOCABULARY_SIZE**: Input vocabulary size. Set to 0 for using all, otherwise it will be truncated to these most frequent words.
   * **MIN_OCCURRENCES_INPUT_VOCAB**: Discard all input words with a frequency below this threshold.
   * **MAX_INPUT_TEXT_LEN**: Maximum length of the input sentences.

   #### Output text parameters
   * **INPUT_VOCABULARY_SIZE: Output vocabulary size. Set to 0 for using all, otherwise it will be truncated to these most frequent words.
   * **MIN_OCCURRENCES_OUTPUT_VOCAB: Discard all output words with a frequency below this threshold.
   * **MAX_INPUT_TEXT_LEN: Maximum length of the output sentences.
   * **MAX_OUTPUT_TEXT_LEN_TEST**: Maximum length of the output sequence during test time.

   #### Optimizer parameters
   * **LOSS**: Loss function to optimize.
   * **CLASSIFIER_ACTIVATION**: Last layer activation function.
   * **OPTIMIZER**: Optimizer to use. See the [available Keras optimizers](https://github.com/MarcBS/keras/blob/master/keras/optimizers.py).
   * **LR**: Learning rate. 
   * **CLIP_C**: During training, clip L2 norm of gradients to this value.
   * **CLIP_V**: During training, clip absolute value of gradients to this value.
   * **SAMPLE_WEIGHTS**: Apply a mask to the output sequence. Should be set to True.
   * **LR_DECAY**: Reduce the learning rate each this number of epochs. Set to None if don't want to decay the learning rate
   * **LR_GAMMA**: Decay rate.

   #### Training parameters
   * **MAX_EPOCH**: Stop when computed this number of epochs.
   * **BATCH_SIZE**: Size of each minibatch.
   * **HOMOGENEOUS_BATCHES**: If activated, use batches with similar output lengths, in order to better profit parallel computations.
   * **JOINT_BATCHES**: When using homogeneous batches, size of the maxibatch.
   * **PARALLEL_LOADERS**: Parallel CPU data batch loaders.
   * **EPOCHS_FOR_SAVE**: Save model each this number of epochs.
   * **WRITE_VALID_SAMPLES**: Write validation samples in file.
   * **SAVE_EACH_EVALUATION**: Save the model each time we evaluate.

   #### Early stop parameters
   * **EARLY_STOP** = Turns on/off the early stop regularizer.
   * **PATIENCE**: We'll stop if we don't improve after this number of evaluations
   * **STOP_METRIC**: Stopping metric.

   #### Model parameters
   * **MODEL_TYPE**: Model to train. See the [model zoo](https://github.com/lvapeab/nmt-keras/blob/master/model_zoo.py) for the supported architectures.
   * **RNN_TYPE**: RNN unit type ('LSTM' and 'GRU' supported).
   * **INIT_FUNCTION**: Initialization function for matrices (see [keras/initializations](https://github.com/MarcBS/keras/blob/master/keras/initializations.py))
   
   ##### Source word embedding configuration
   * **SOURCE_TEXT_EMBEDDING_SIZE**: Source language word embedding size.
   * **SRC_PRETRAINED_VECTORS**: Path to source pretrained vectors. See the [utils](https://github.com/lvapeab/nmt-keras/tree/master/utils) folder for preprocessing scripts. Set to None if you don't want to use source pretrained vectors. When using pretrained word embeddings. this parameter must match with the source word embeddings size
   * **SRC_PRETRAINED_VECTORS_TRAINABLE**: Finetune or not the target word embedding vectors.

   ##### Target word embedding configuration
   * **TARGET_TEXT_EMBEDDING_SIZE**: Source language word embedding size.
   * **TRG_PRETRAINED_VECTORS**: Path to target pretrained vectors. See the [utils](https://github.com/lvapeab/nmt-keras/tree/master/utils) folder for preprocessing scripts. Set to None if you don't want to use source pretrained vectors. When using pretrained word embeddings. this parameter must match with the target word embeddings size
   * **TRG_PRETRAINED_VECTORS_TRAINABLE**: Finetune or not the target word embedding vectors.

   ##### Encoder configuration
   * **ENCODER_HIDDEN_SIZE**: Encoder RNN size.
   * **BIDIRECTIONAL_ENCODER**: Use a bidirectional encoder.
   * **N_LAYERS_ENCODER**: Stack this number of encoding layers
   * **BIDIRECTIONAL_DEEP_ENCODER**: Use bidirectional encoder in all stacked encoding layers

   ##### Decoder configuration

   * **DECODER_HIDDEN_SIZE**: Decoder RNN size.
   * **N_LAYERS_DECODER**: Stack this number of decoding layers.
   * **ADDITIONAL_OUTPUT_MERGE_MODE**: Merge mode for the [deep output layer](https://arxiv.org/abs/1312.6026).
   * **SKIP_VECTORS_HIDDEN_SIZE**: Deep output layer size
   * **INIT_LAYERS**: Initialize the first decoder state with these layers (from the encoder).
   * **DEEP_OUTPUT_LAYERS**: Additional Fully-Connected layers applied before softmax.

   #### Regularizers
   
   * **WEIGHT_DECAY**: L2 regularization in non-recurrent layers.
   * **RECURRENT_WEIGHT_DECAY**: L2 regularization in recurrent layers
   * **USE_DROPOUT**: Use dropout in non-recurrent layers.
   * **DROPOUT_P**: Percentage of units to drop in non-recurrent layers

   * **USE_RECURRENT_DROPOUT**: Use dropout in recurrent layers.
   * **RECURRENT_DROPOUT_P**: Percentage of recurrent units to drop.

   * **USE_NOISE**: Apply gaussian noise during training.
   * **NOISE_AMOUNT**: Amount of noise.

   * **USE_BATCH_NORMALIZATION**:  Batch normalization regularization in non-recurrent layers and recurrent inputs. If True it is recommended to deactivate Dropout.
   * **BATCH_NORMALIZATION_MODE**: Sample-wise or feature-wise BN mode. 

   * **USE_PRELU**: Apply use PReLU activations as regularizer.
   * **USE_L2**: Apply L2 function on the features.

   #### Storage and plotting parameters

   * **MODEL_NAME**: Name for the model.
   * **EXTRA_NAME**: MODEL_NAME suffix
   * **STORE_PATH**: Models and evaluation results will be stored here.
   * **DATASET_STORE_PATH**: Dataset instance will be stored here.

   * **SAMPLING_SAVE_MODE**: Save evaluation outputs in this format. Set to 'list' for a raw file.
   * **VERBOSE**: Verbosity level.
   * **RELOAD**: Reload a stored model. If 0 start training from scratch, otherwise use the model from this epoch/update.
   * **REBUILD_DATASET**: Build dataset again or use a stored instance.
   * **MODE**: 'training' or 'sampling' (if 'sampling' then RELOAD must be greater than 0 and EVAL_ON_SETS will be used). For 'sampling' mode, is recommended to use the [sample_ensemble](https://github.com/lvapeab/nmt-keras/blob/master/examples/documentation/ensembling_tutorial.md) script.

