# NMT-Keras


Neural Machine Translation with Keras (+ Theano backend)

## Features (in addition to the full Keras cosmos): 

 * Attention model over the input sequence of annotations.
 * Peeked decoder: The previously generated word is an input of the current timestep.
 * MLPs for initializing the RNN hidden and memory state.
 * Support for GRU/LSTM networks.
 * Multilayered residual GRU/LSTM networks (WIP).
 * Beam search decoding.
 * Ensemble decoding.
 * [Spearmint] (https://github.com/HIPS/Spearmint) [wrapper](https://github.com/lvapeab/nmt-keras/tree/master/meta-optimizers/spearmint) for hyperparameter optimization

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
 
 4) We can use model ensembles with the [sample_ensemble.py](https://github.com/lvapeab/nmt-keras/blob/master/sample_ensemble.py) script. For example, if we want to use the models from the first three epochs on the val split, just run:
 ```bash
  python sample_ensemble.py --models trained_models/tutorial_model/epoch_1  trained_models/tutorial_model/epoch_2 -ds datasets/Dataset_tutorial_dataset.pkl -s val
  ```
 
 


## Resources

 * In [examples/documentation/neural_machine_translation.pdf](https://github.com/lvapeab/nmt-keras/blob/master/examples/documentation/neural_machine_translation.pdf) you'll find an overview of an attentional NMT system.

 * In [examples/](https://github.com/lvapeab/nmt-keras/blob/master/examples/) you'll find some tutorials for running this library. They are expected to be followed in order:
    
    1) [Dataset set up](https://github.com/lvapeab/nmt-keras/blob/master/examples/1_dataset_tutorial.ipynb): Shows how to invoke and configure a Dataset instance for a translation problem.
    
    2) [Training tutorial](https://github.com/lvapeab/nmt-keras/blob/master/examples/2_training_tutorial.ipynb): Shows how to call a translation model, link it with the dataset object and construct calllbacks for monitorizing the training. 
    
    3) [Decoding tutorial](https://github.com/lvapeab/nmt-keras/blob/master/examples/3_decoding_tutorial.ipynb): Shows how to call a trained translation model and use it to translate new text. 

    4) [NMT model tutorial](https://github.com/lvapeab/nmt-keras/blob/master/examples/4_nmt_model_tutorial.ipynb): Shows how to build a state-of-the-art NMT model with Keras in few (~50) lines. 


## Contact

Álvaro Peris ([web page](http://lvapeab.github.io/)): lvapeab@prhlt.upv.es 

## Acknowledgement

Much of this library has been developed together with [Marc Bolaños](https://github.com/MarcBS) ([web page](http://www.ub.edu/cvub/marcbolanos/)) for other sequence-to-sequence problems. 

To see other projects following the philosophy of NMT-Keras, take a look here:
 
[VIBIKNet for Visual Question Answering](https://github.com/MarcBS/VIBIKNet)

[ABiViRNet for Video Description](https://github.com/lvapeab/ABiViRNet)

## Warning 

NMT-Keras is under development. There are many features still unimplemented/possibly buggy. If you find a bug or desire a specific feature, please do not hesitate to contact me.