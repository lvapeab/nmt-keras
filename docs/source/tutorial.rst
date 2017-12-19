#########
Tutorials
#########

This page contains some examples and tutorials showing how the library works. All tutorials have an `iPython notebook version`_.

.. _iPython notebook version: https://github.com/lvapeab/nmt-keras/blob/master/examples

Almost every variable tutorials representing model hyperparameters have been intentionally hardcoded in the tutorials,
aiming to facilitate readability. On a real execution, these values are taken from the `config.py` file.

All tutorials have been executed from the root `nmt-keras` folder. These tutorials basically are a split version of the execution pipeline of the library. If you run `python main.py`, you'll execute almost the same as tutorials 1, 2 and 4.

The translation task is *EuTrans* (`Amengual et al.`_), a toy-task mainly used for debugging purposes.

.. _Amengual et al.: http://link.springer.com/article/10.1023/A:1011116115948

****************
Dataset tutorial
****************

First, we'll create a Dataset_ instance, in order to properly manage the data. First, we are creating a Dataset_ object (from the `Multimodal Keras Wrapper`_ library).
Let's make some imports and create an empty Dataset_ instance::

    from keras_wrapper.dataset import Dataset, saveDataset
    from data_engine.prepare_data import keep_n_captions
    ds = Dataset('tutorial_dataset', 'tutorial', silence=False)

.. _Multimodal Keras Wrapper: https://github.com/lvapeab/multimodal_keras_wrapper
.. _Dataset: http://marcbs.github.io/multimodal_keras_wrapper/tutorial.html#basic-components


Now that we have the empty Dataset_, we must indicate its inputs and outputs. In our case, we'll have two different inputs and one single output:

1. Outputs::
    **target_text**: Sentences in the target language.

2. Inputs::
    **source_text**: Sentences in the source language.

    **state_below**: Sentences in the target language, but shifted one position to the right (for teacher-forcing training of the model).

For setting up the outputs, we use the setOutputs function, with the appropriate parameters. Note that, when we are building the dataset for the training split, we build the vocabulary (up to 30000 words)::

    ds.setOutput('examples/EuTrans/training.en',
                 'train',
                 type='text',
                 id='target_text',
                 tokenization='tokenize_none',
                 build_vocabulary=True,
                 pad_on_batch=True,
                 sample_weights=True,
                 max_text_len=30,
                 max_words=30000,
                 min_occ=0)

    ds.setOutput('examples/EuTrans/dev.en',
                 'val',
                 type='text',
                 id='target_text',
                 pad_on_batch=True,
                 tokenization='tokenize_none',
                 sample_weights=True,
                 max_text_len=30,
                 max_words=0)

Similarly, we introduce the source text data, with the setInputs function. Again, when building the training split, we must construct the vocabulary::



    ds.setInput('examples/EuTrans/training.es',
                'train',
                type='text',
                id='source_text',
                pad_on_batch=True,
                tokenization='tokenize_none',
                build_vocabulary=True,
                fill='end',
                max_text_len=30,
                max_words=30000,
                min_occ=0)
    ds.setInput('examples/EuTrans/dev.es',
                'val',
                type='text',
                id='source_text',
                pad_on_batch=True,
                tokenization='tokenize_none',
                fill='end',
                max_text_len=30,
                min_occ=0)




...and for the `state_below` data. Note that: 1) The offset flat is set to 1, which means that the text will be shifted to the right 1 position. 2) During sampling time, we won't have this input. Hence, we 'hack' the dataset model by inserting an artificial input, of type 'ghost' for the validation split::

    ds.setInput('examples/EuTrans/training.en',
                'train',
                type='text',
                id='state_below',
                required=False,
                tokenization='tokenize_none',
                pad_on_batch=True,
                build_vocabulary='target_text',
                offset=1,
                fill='end',
                max_text_len=30,
                max_words=30000)
    ds.setInput(None,
                'val',
                type='ghost',
                id='state_below',
                required=False)


Next, we match the references with the inputs, in order to evaluate against the raw references::

    keep_n_captions(ds, repeat=1, n=1, set_names=['val'])


Finally, we can save our dataset instance for using it in other experiments::

    saveDataset(ds, 'datasets')


*****************
Training tutorial
*****************
Now, we'll create and train a Neural Machine Translation (NMT) model.
We'll build the so-called `GroundHogModel`. It is defined at the `model_zoo.py` file.
If you followed prior tutorial, you should have a dataset instance. Otherwise, you should follow that notebook first.

So, let's go! First, we make some imports, load the default parameters and the dataset::

    from config import load_parameters
    from model_zoo import TranslationModel
    import utils
    from keras_wrapper.cnn_model import loadModel
    from keras_wrapper.dataset import loadDataset
    params = load_parameters()
    dataset = loadDataset('datasets/Dataset_tutorial_dataset.pkl')

Since the number of words in the dataset may be unknown beforehand, we must update the params information according to the dataset instance::


    params['INPUT_VOCABULARY_SIZE'] = dataset.vocabulary_len['source_text']
    params['OUTPUT_VOCABULARY_SIZE'] = dataset.vocabulary_len['target_text']

Now, we create a `TranslationModel` object: An instance of a `Model_Wrapper`_ object from `Multimodal Keras Wrapper`_.
We specify the type of the model (`GroundHogModel`) and the vocabularies from the Dataset_::

    nmt_model = TranslationModel(params,
                                 model_type='GroundHogModel',
                                 model_name='tutorial_model',
                                 vocabularies=dataset.vocabulary,
                                 store_path='trained_models/tutorial_model/',
                                 verbose=True)

.. _Model_Wrapper: http://marcbs.github.io/multimodal_keras_wrapper/tutorial.html#basic-components

Now, we must define the inputs and outputs mapping from our Dataset instance to our model::

    inputMapping = dict()
    for i, id_in in enumerate(params['INPUTS_IDS_DATASET']):
        pos_source = dataset.ids_inputs.index(id_in)
        id_dest = nmt_model.ids_inputs[i]
        inputMapping[id_dest] = pos_source
    nmt_model.setInputsMapping(inputMapping)

    outputMapping = dict()
    for i, id_out in enumerate(params['OUTPUTS_IDS_DATASET']):
        pos_target = dataset.ids_outputs.index(id_out)
        id_dest = nmt_model.ids_outputs[i]
        outputMapping[id_dest] = pos_target
    nmt_model.setOutputsMapping(outputMapping)




We can add some callbacks for controlling the training (e.g. Sampling each N updates, early stop, learning rate annealing...).
For instance, let's build a `PrintPerformanceMetricOnEpochEndOrEachNUpdates` callback. Each 2 epochs, it will compute the 'coco' scores on the development set.
We need to pass some variables to the callback (in the extra_vars dictionary)::

    from keras_wrapper.extra.callbacks import *
    extra_vars = {'language': 'en',
                  'n_parallel_loaders': 8,
                  'tokenize_f': eval('dataset.' + 'tokenize_none'),
                  'beam_size': 12,
                  'maxlen': 50,
                  'model_inputs': ['source_text', 'state_below'],
                  'model_outputs': ['target_text'],
                  'dataset_inputs': ['source_text', 'state_below'],
                  'dataset_outputs': ['target_text'],
                  'normalize': True,
                  'alpha_factor': 0.6,
                  'val':{'references': dataset.extra_variables['val']['target_text']}
                  }
    vocab = dataset.vocabulary['target_text']['idx2words']
    callbacks = []
    callbacks.append(PrintPerformanceMetricOnEpochEnd(nmt_model,
                                                      dataset,
                                                      gt_id='target_text',
                                                      metric_name=['coco'],
                                                      set_name=['val'],
                                                      batch_size=50,
                                                      each_n_epochs=2,
                                                      extra_vars=extra_vars,
                                                      reload_epoch=0,
                                                      is_text=True,
                                                      index2word_y=vocab,
                                                      sampling_type='max_likelihood',
                                                      beam_search=True,
                                                      save_path=nmt_model.model_path,
                                                      start_eval_on_epoch=0,
                                                      write_samples=True,
                                                      write_type='list',
                                                      save_each_evaluation=True,
                                                      verbose=True))


Now we are almost ready to train. We set up some training parameters...::

    training_params = {'n_epochs': 100,
                       'batch_size': 40,
                       'maxlen': 30,
                       'epochs_for_save': 1,
                       'verbose': 0,
                       'eval_on_sets': [],
                       'n_parallel_loaders': 8,
                       'extra_callbacks': callbacks,
                       'reload_epoch': 0,
                       'epoch_offset': 0}



And train!::

    nmt_model.trainNet(dataset, training_params)


For a description of the training output, refer to the `typical output`_ document.

.. _typical output: https://github.com/lvapeab/nmt-keras/blob/master/examples/documentation/typical_output.md

*****************
Decoding tutorial
*****************


Now, we'll load from disk a trained Neural Machine Translation (NMT) model and we'll apply it for translating new text. This is done by the sample_ensemble_ script.

This tutorial assumes that you followed both previous tutorials. In this case, we want to translate the 'test' split of our dataset.

As before, let's import some stuff and load the dataset instance::

    from config import load_parameters
    from data_engine.prepare_data import keep_n_captions
    from keras_wrapper.cnn_model import loadModel
    from keras_wrapper.dataset import loadDataset
    params = load_parameters()
    dataset = loadDataset('datasets/Dataset_tutorial_dataset.pkl')


Since we want to translate a new data split ('test') we must add it to the dataset instance, just as we did before (at the first tutorial).
In case we also had the refences of the test split and we wanted to evaluate it, we can add it to the dataset. Note that this is not mandatory and we could just predict without evaluating.::

    dataset.setInput('examples/EuTrans/test.es',
                'test',
                type='text',
                id='source_text',
                pad_on_batch=True,
                tokenization='tokenize_none',
                fill='end',
                max_text_len=100,
                min_occ=0)

    dataset.setInput(None,
    'test',
    type='ghost',
    id='state_below',
    required=False)

.. _sample_ensemble: https://github.com/lvapeab/nmt-keras/blob/master/examples/documentation/ensembling_tutorial.md


Now, let's load the translation model. Suppose we want to load the model saved at the end of the epoch 4::

    params['INPUT_VOCABULARY_SIZE'] = dataset.vocabulary_len[params['INPUTS_IDS_DATASET'][0]]
    params['OUTPUT_VOCABULARY_SIZE'] = dataset.vocabulary_len[params['OUTPUTS_IDS_DATASET'][0]]
    # Load model
    nmt_model = loadModel('trained_models/tutorial_model', 4)


Once we loaded the model, we just have to invoke the sampling method (in this case, the Beam Search algorithm) for the 'test' split::

    params_prediction = {'max_batch_size': 50,
                         'n_parallel_loaders': 8,
                         'predict_on_sets': ['test'],
                         'beam_size': 12,
                         'maxlen': 50,
                         'model_inputs': ['source_text', 'state_below'],
                         'model_outputs': ['target_text'],
                         'dataset_inputs': ['source_text', 'state_below'],
                         'dataset_outputs': ['target_text'],
                         'normalize': True,
                         'alpha_factor': 0.6
                         }
    predictions = nmt_model.predictBeamSearchNet(dataset, params_prediction)['test']


Up to this moment, in the variable 'predictions', we have the indices of the words of the hypotheses. We must decode them into words. For doing this, we'll use the dictionary stored in the dataset object::

    from keras_wrapper.utils import decode_predictions_beam_search
    vocab = dataset.vocabulary['target_text']['idx2words']
    predictions = decode_predictions_beam_search(predictions,
                                                 vocab,
                                                 verbose=params['VERBOSE'])

Finally, we store the system hypotheses::

    filepath = nmt_model.model_path+'/' + 'test' + '_sampling.pred'  # results file
    from keras_wrapper.extra.read_write import list2file
    list2file(filepath, predictions)




If we have the references of this split, we can also evaluate the performance of our system on it. First, we must add them to the dataset object::

    # In case we had the references of this split, we could also load the split and evaluate on it
    dataset.setOutput('examples/EuTrans/test.en',
                 'test',
                 type='text',
                 id='target_text',
                 pad_on_batch=True,
                 tokenization='tokenize_none',
                 sample_weights=True,
                 max_text_len=30,
                 max_words=0)
    keep_n_captions(dataset, repeat=1, n=1, set_names=['test'])



Next, we call the evaluation system: The Coco-caption_ package. Although its main usage is for multimodal captioning, we can use it in machine translation::


    from keras_wrapper.extra import evaluation
    metric = 'coco'
    # Apply sampling
    extra_vars = dict()
    extra_vars['tokenize_f'] = eval('dataset.' + 'tokenize_none')
    extra_vars['language'] = params['TRG_LAN']
    extra_vars['test'] = dict()
    extra_vars['test']['references'] = dataset.extra_variables['test']['target_text']
    metrics = evaluation.select[metric](pred_list=predictions,
                                        verbose=1,
                                        extra_vars=extra_vars,
                                        split='test')

.. _Coco-caption: https://github.com/lvapeab/coco-caption


******************
NMT model tutorial
******************



In this module, we are going to create an encoder-decoder model with:

    * A bidirectional GRU encoder and a GRU decoder
    * An attention model
    * The previously generated word feeds back de decoder
    * MLPs for initializing the initial RNN state
    * Skip connections from inputs to outputs
    * Beam search.

As usual, first we import the necessary stuff::

    from keras.layers import *
    from keras.models import model_from_json, Model
    from keras.optimizers import Adam, RMSprop, Nadam, Adadelta, SGD, Adagrad, Adamax
    from keras.regularizers import l2
    from keras_wrapper.cnn_model import Model_Wrapper
    from keras_wrapper.extra.regularize import Regularize

And define the dimesnions of our model. For instance, a word embedding size of 50 and 100 units in RNNs.
The inputs/outpus are defined as in previous tutorials.::

    ids_inputs = ['source_text', 'state_below']
    ids_outputs = ['target_text']
    word_embedding_size = 50
    hidden_state_size = 100
    input_vocabulary_size=686  # Autoset in the library
    output_vocabulary_size=513  # Autoset in the library

Now, let's define our encoder. First, we have to create an Input layer to connect the input text to our model.
Next, we'll apply a word embedding to the sequence of input indices. This word embedding will feed a Bidirectional GRU network, which will produce our sequence of annotations::

    # 1. Source text input
    src_text = Input(name=ids_inputs[0],
                     batch_shape=tuple([None, None]), # Since the input sequences have variable-length, we do not retrict the Input shape
                     dtype='int32')
    # 2. Encoder
    # 2.1. Source word embedding
    src_embedding = Embedding(input_vocabulary_size, word_embedding_size,
                              name='source_word_embedding', mask_zero=True # Zeroes as mask
                              )(src_text)
    # 2.2. BRNN encoder (GRU/LSTM)
    annotations = Bidirectional(GRU(hidden_state_size,
                                    return_sequences=True  # Return the full sequence
                                    ),
                                name='bidirectional_encoder',
                                merge_mode='concat')(src_embedding)



Once we have built the encoder, let's build our decoder.
First, we have an additional input: The previously generated word (the so-called state_below). We introduce it by means of an Input layer and a (target language) word embedding::

    # 3. Decoder
    # 3.1.1. Previously generated words as inputs for training -> Teacher forcing
    next_words = Input(name=ids_inputs[1], batch_shape=tuple([None, None]), dtype='int32')
    # 3.1.2. Target word embedding
    state_below = Embedding(output_vocabulary_size, word_embedding_size,
                            name='target_word_embedding',
                            mask_zero=True)(next_words)



The initial hidden state of the decoder's GRU is initialized by means of a MLP (in this case, single-layered) from the average of the annotations. We also aplly the mask to the annotations::


    ctx_mean = MaskedMean()(annotations)
    annotations = MaskLayer()(annotations)  # We may want the padded annotations
    initial_state = Dense(hidden_state_size, name='initial_state',
                          activation='tanh')(ctx_mean)

So, we have the input of our decoder::

    input_attentional_decoder = [state_below, annotations, initial_state]



Note that, for a sample, the sequence of annotations and initial state is the same, independently of the decoding time-step.
In order to avoid computation time, we build two models, one for training and the other one for sampling.
They will share weights, but the sampling model will be made up of two different models. One (model_init) will compute the sequence of annotations and initial_state.
The other model (model_next) will compute a single recurrent step, given the sequence of annotations, the previous hidden state and the generated words up to this moment.

Therefore, now we slightly change the form of declaring layers. We must share layers between the decoding models.

So, let's start by building the attentional-conditional GRU::

    # Define the AttGRUCond function
    sharedAttGRUCond = AttGRUCond(hidden_state_size,
                                  return_sequences=True,
                                  return_extra_variables=True, # Return attended input and attenton weights
                                  return_states=True # Returns the sequence of hidden states (see discussion above)
                                  )
    [proj_h, x_att, alphas, h_state] = sharedAttGRUCond(input_attentional_decoder) # Apply shared_AttnGRUCond to our input

Now, we set skip connections between input and output layer. Note that, since we have a temporal dimension because of the RNN decoder, we must apply the layers in a TimeDistributed way.
Finally, we will merge all skip-connections and apply a 'tanh' no-linearlity::

    # Define layer function
    shared_FC_mlp = TimeDistributed(Dense(word_embedding_size, activation='linear',),
                                    name='logit_lstm')
    # Apply layer function
    out_layer_mlp = shared_FC_mlp(proj_h)

    # Define layer function
    shared_FC_ctx = TimeDistributed(Dense(word_embedding_size, activation='linear'),
                                    name='logit_ctx')
    # Apply layer function
    out_layer_ctx = shared_FC_ctx(x_att)
    shared_Lambda_Permute = PermuteGeneral((1, 0, 2))
    out_layer_ctx = shared_Lambda_Permute(out_layer_ctx)

    # Define layer function
    shared_FC_emb = TimeDistributed(Dense(word_embedding_size, activation='linear'),
                                    name='logit_emb')
    # Apply layer function
    out_layer_emb = shared_FC_emb(state_below)

    additional_output = merge([out_layer_mlp, out_layer_ctx, out_layer_emb], mode='sum', name='additional_input')
    shared_activation_tanh = Activation('tanh')
    out_layer = shared_activation_tanh(additional_output)

Now, we'll' apply a deep output layer, with Maxout activation::

    shared_maxout = TimeDistributed(MaxoutDense(word_embedding_size), name='maxout_layer')
    out_layer = shared_maxout(out_layer)


Finally, we apply a softmax function for obtaining a probability distribution over the target vocabulary words at each timestep::

    shared_FC_soft = TimeDistributed(Dense(output_vocabulary_size,
                                                   activation='softmax',
                                                   name='softmax_layer'),
                                             name=ids_outputs[0])
    softout = shared_FC_soft(out_layer)

That's all! We built a NMT Model!

NMT models for decoding
=======================

Now, let's build the models required for sampling. Recall that we are building two models, one for encoding the inputs and the other one for advancing steps in the decoding stage.

Let's start with model_init. It will take the usual inputs (src_text and state_below) and will output:

1. The vector probabilities (for timestep 1).
2. The sequence of annotations (from encoder).
3. The current decoder's hidden state.

The only restriction here is that the first output must be the output layer (probabilities) of the model.::

    model_init = Model(input=[src_text, next_words], output=[softout, annotations, h_state])
    # Store inputs and outputs names for model_init
    ids_inputs_init = ids_inputs

    # first output must be the output probs.
    ids_outputs_init = ids_outputs + ['preprocessed_input', 'next_state']



Next, we will be the model_next. It will have the following inputs:

    * Preprocessed input
    * Previously generated word
    * Previous hidden state

And the following outputs:

    * Model probabilities
    * Current hidden state

First, we define the inputs::

    preprocessed_size = hidden_state_size*2 # Because we have a bidirectional encoder
    preprocessed_annotations = Input(name='preprocessed_input', shape=tuple([None, preprocessed_size]))
    prev_h_state = Input(name='prev_state', shape=tuple([hidden_state_size]))
    input_attentional_decoder = [state_below, preprocessed_annotations, prev_h_state]


And now, we build the model, using the functions stored in the 'shared*' variables declared before::

    # Apply decoder
    [proj_h, x_att, alphas, h_state] = sharedAttGRUCond(input_attentional_decoder)
    out_layer_mlp = shared_FC_mlp(proj_h)
    out_layer_ctx = shared_FC_ctx(x_att)
    out_layer_ctx = shared_Lambda_Permute(out_layer_ctx)
    out_layer_emb = shared_FC_emb(state_below)
    additional_output = merge([out_layer_mlp, out_layer_ctx, out_layer_emb], mode='sum', name='additional_input')
    out_layer = shared_activation_tanh(additional_output)
    out_layer = shared_maxout(out_layer)
    softout = shared_FC_soft(out_layer)
    model_next = Model(input=[next_words, preprocessed_annotations, prev_h_state],
                       output=[softout, preprocessed_annotations, h_state])

Finally, we store inputs/outputs for model_next. In addition, we create a couple of dictionaries, matching inputs/outputs from the different models (model_init->model_next, model_nex->model_next)::

    # Store inputs and outputs names for model_next
    # first input must be previous word
    ids_inputs_next = [ids_inputs[1]] + ['preprocessed_input', 'prev_state']
    # first output must be the output probs.
    ids_outputs_next = ids_outputs + ['preprocessed_input', 'next_state']

    # Input -> Output matchings from model_init to model_next and from model_next to model_nextxt
    matchings_init_to_next = {'preprocessed_input': 'preprocessed_input', 'next_state': 'prev_state'}
    matchings_next_to_next = {'preprocessed_input': 'preprocessed_input', 'next_state': 'prev_state'}




And that's all! For using this model together with the facilities provided by the staged_model_wrapper library, we should declare the model as a method of a Model_Wrapper class.
A complete example of this with additional features can be found at model_zoo.py_.


.. _model_zoo.py: https://github.com/lvapeab/nmt-keras/blob/master/model_zoo.py
