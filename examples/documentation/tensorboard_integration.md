# Tensorboard integration

[TensorBoard](https://www.tensorflow.org/get_started/summaries_and_tensorboard) is a visualization tool provided with TensorFlow.

It can be accessed by NMT-Keras and provide visualization of the learning process, dynamic graphs of our training and metrics, as well representation of different layers (such as word embeddings). Of course, this tool is only available with the Tensorflow backend. 

In this document, we'll set some parameters and explore some of the options that Tensorboard provides. We'll:

    * Configure Tensorboard and NMT-Keras.
    * Visualize the learning process (loss curves).
    * Visualize the computation graphs built by NMT Keras.
    * Visualize the words embeddings obtained during the training stage.
   
   
In the [configuration file](https://github.com/lvapeab/nmt-keras/blob/master/config.py) we have available the following tensorboard-related options:
 
```python
TENSORBOARD = True                       # Switches On/Off the tensorboard callback
LOG_DIR = 'tensorboard_logs'             # Directory to store teh model. Will be created inside STORE_PATH
EMBEDDINGS_FREQ = 1                      # Frequency (in epochs) at which selected embedding layers will be saved.
EMBEDDINGS_LAYER_NAMES = [               # A list of names of layers to keep eye on. If None or empty list all the embedding layer will be watched.
'source_word_embedding',
'target_word_embedding']
EMBEDDINGS_METADATA = None               # Dictionary which maps layer name to a file name in which metadata for this embedding layer is saved.
LABEL_WORD_EMBEDDINGS_WITH_VOCAB = True  # Whether to use vocabularies as word embeddings labels (will overwrite EMBEDDINGS_METADATA)
WORD_EMBEDDINGS_LABELS = [               # Vocabularies for labeling. Must match EMBEDDINGS_LAYER_NAMES
                         'source_text',
                         'target_text']
```

With these options, we are telling Tensorboard where to store the data we want to visualize: loss curve, computation graph and word embeddings. 
Moreover, we are specifying the word embedding layers that we want to visualize. By setting the `WORD_EMBEDDINGS_LABELS` to the corresponding `Dataset` ids, 
we can print labels in the word embedding visualization. 


Now, we run a regular training: ``python main.py``. If we `cd` to the model directory, we'll see a directiory named `tensorboard_logs`. Now, we launch Tensorboard on this directory:
 
```bash
$ tensorboard --logdir=tensorboard_logs
TensorBoard 0.1.5 at http://localhost:6006 (Press CTRL+C to quit) 
```

We can open Tensorboard in our browser (http://localhost:6006) and track NMT-Keras information:


### Loss curve
 
 <div align="left">
  <br><br><img src="https://raw.githubusercontent.com/lvapeab/nmt-keras/master/examples/documentation/imgs/tb-scalar.png"><br><br>
</div>

    
    
### Model graphs

 <div align="left">
  <br><br><img src="https://raw.githubusercontent.com/lvapeab/nmt-keras/master/examples/documentation/imgs/tb-graph.png"><br><br>
</div>


### Embedding visualization

 <div align="left">
  <br><br><img src="https://raw.githubusercontent.com/lvapeab/nmt-keras/master/examples/documentation/imgs/tb-embeddings.png"><br><br>
</div>
