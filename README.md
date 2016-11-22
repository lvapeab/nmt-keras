# nmt-keras


Neural Machine Translation with Theano&amp;Keras

## Features: 

 * Attention model over the input sequence of annotations
 * Peeked decoder: The previously generated word is an input of the current timestep
 * MLPs for initializing the RNN hidden and memory state
 * Support for GRU/LSTM networks
 * Beam search decoding

## Requirements

NMT-Keras requires the following libraries:

 - [Our version of Keras](https://github.com/MarcBS/keras) v1.2
 - [Staged Keras Wrapper](https://github.com/MarcBS/staged_keras_wrapper) v0.1 or newer
 - [Coco-caption evaluation package](https://github.com/lvapeab/coco-caption/tree/master/pycocoevalcap/)

## Instructions:

Assuming you have a dataset and features extracted from the video frames:
 
1) Prepare the inputs/outputs of your model in `data_engine/prepare_data.py`
  
2) Set a model configuration in  `config.py` 
 
2) Train!:

  ``
 python main.py
 ``

## Contact

Álvaro Peris ([web page](http://lvapeab.github.io/)): lvapeab@prhlt.upv.es 

