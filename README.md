# NMT-Keras


Neural Machine Translation with Keras (+ Theano backend)

## Features (in addition to the full Keras cosmos): 

 * Attention model over the input sequence of annotations.
 * Peeked decoder: The previously generated word is an input of the current timestep.
 * MLPs for initializing the RNN hidden and memory state.
 * Support for GRU/LSTM networks.
 * Multilayered residual GRU/LSTM networks (WIP).
 * Beam search decoding.

## Requirements

NMT-Keras requires the following libraries:

 - [Our version of Keras](https://github.com/MarcBS/keras)
 - [Staged Keras Wrapper](https://github.com/MarcBS/staged_keras_wrapper) ([Documentation](http://marcbs.github.io/staged_keras_wrapper/))
 - [Coco-caption evaluation package](https://github.com/lvapeab/coco-caption/tree/master/pycocoevalcap/)


## Instructions

1) Set a model configuration in `config.py`. Each parameter is commented.

2) Train!:

  ``
 python main.py
 ``

3) For evaluating on a new partition, we just need to adequately modify [config.py](https://github.com/lvapeab/nmt-keras/blob/master/config.py). For example, if we want to obtain the translations of the test set, using the model obtained at the end of the epoch 5, we should do:
 ```python
 - MODE = 'sampling'
 - RELOAD = 5
 - EVAL_ON_SETS = ['test']
  ```
 - Run `python main.py`


## Resources

 * In [examples/neural_machine_translation.pdf](https://github.com/lvapeab/nmt-keras/blob/master/examples/neural_machine_translation.pdf) you'll find an overview of an attentional NMT system.

 * In [examples/*.ipynb](https://github.com/lvapeab/nmt-keras/blob/master/examples/) you'll find some tutorials for running this library.

## Contact

Álvaro Peris ([web page](http://lvapeab.github.io/)): lvapeab@prhlt.upv.es 

## Acknowledgement

Much of this library has been developed together with [Marc Bolaños](https://github.com/MarcBS) ([web page](http://www.ub.edu/cvub/marcbolanos/)) for other sequence-to-sequence problems.

## Warning 

NMT-Keras is under development. There are many features still unimplemented/possibly buggy. If you find a bug or desire a specific feature, please do not hesitate to contact me.