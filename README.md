# NMT-Keras


Neural Machine Translation with Keras (+ Theano backend)

## Features: 

 * Attention model over the input sequence of annotations.
 * Peeked decoder: The previously generated word is an input of the current timestep.
 * MLPs for initializing the RNN hidden and memory state.
 * Support for GRU/LSTM networks.
 * Multilayered residual GRU/LSTM networks (WIP).
 * Beam search decoding.

## Requirements

NMT-Keras requires the following libraries:

 - [Our version of Keras](https://github.com/MarcBS/keras)
 - [Staged Keras Wrapper](https://github.com/MarcBS/staged_keras_wrapper) 
 - [Coco-caption evaluation package](https://github.com/lvapeab/coco-caption/tree/master/pycocoevalcap/)

## Instructions

1) Set a model configuration in `config.py`. Each parameter is commented.

2) Train!:

  ``
 python main.py
 ``

3) For evaluating on a new partition, we just need to adequately modify `config.py`:
 - Set the EVAL_ON_SETS variable to the desired one (e.g. test)
 - Set RELOAD = 1
 - Set MODE = 'sampling'
 - Run `python main.py`
 
## Contact

Álvaro Peris ([web page](http://lvapeab.github.io/)): lvapeab@prhlt.upv.es 

## Acknowledgement

Much of this library has been developed together with [Marc Bolaños](https://github.com/MarcBS) ([web page](http://www.ub.edu/cvub/marcbolanos/)) for other sequence-to-sequence problems.

## Warning 

NMT-Keras is under development. There are many features still unimplemented/possibly buggy. If you find a bug or desire a specific feature, please do not hesitate to contact me.