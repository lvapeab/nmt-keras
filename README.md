# NMT-Keras

Neural Machine Translation with Keras (+ Theano backend)


## Features (in addition to the full Keras cosmos): 

 * Beam search decoding.
 * Unknown words replacement (see Section 3.3 from [this paper](https://arxiv.org/pdf/1412.2007v2.pdf))
 * Ensemble decoding ([sample_ensemble.py](https://github.com/lvapeab/nmt-keras/blob/master/sample_ensemble.py)).
 * Attention model over the input sequence of annotations.
 * Peeked decoder: The previously generated word is an input of the current timestep.
 * Use of pretrained ([Glove](http://nlp.stanford.edu/projects/glove/) or [Word2Vec](https://code.google.com/archive/p/word2vec/)) word embedding vectors.
 * MLPs for initializing the RNN hidden and memory state.
 * Support for GRU/LSTM networks.
 * Multilayered residual GRU/LSTM networks.
 * N-best list generation (as byproduct of the beam search process).
 * [Spearmint](https://github.com/HIPS/Spearmint) [wrapper](https://github.com/lvapeab/nmt-keras/tree/master/meta-optimizers/spearmint) for hyperparameter optimization

## Requirements

NMT-Keras requires the following libraries:

 - [Our version of Keras](https://github.com/MarcBS/keras) (Recommended v. 1.2.3 or newer, as it solves some issues)
 - [Staged Keras Wrapper](https://github.com/lvapeab/staged_keras_wrapper) (v. 0.7 or newer) ([Documentation](http://marcbs.github.io/staged_keras_wrapper/) and [tutorial](http://marcbs.github.io/multimodal_keras_wrapper/tutorial.html))
 - [Coco-caption evaluation package](https://github.com/lvapeab/coco-caption/tree/master/pycocoevalcap/) (Only required to perform evaluation)


## Instructions

1) Set a model configuration in `config.py`. Each parameter is commented.

2) Train!:

  ``
 python main.py
 ``

3) We can translate new text using the [sample_ensemble.py](https://github.com/lvapeab/nmt-keras/blob/master/sample_ensemble.py) script. Please, refer to the [ensembling_tutorial](https://github.com/lvapeab/nmt-keras/blob/master/examples/documentation/ensembling_tutorial.md) for more details of this script. 
In short, if we want to use the models from the first three epochs on the val split, just run:
 ```bash
  python sample_ensemble.py --models trained_models/tutorial_model/epoch_1  trained_models/tutorial_model/epoch_2 -ds datasets/Dataset_tutorial_dataset.pkl -t text_to_translate
  ```
 
 
* The [score.py](https://github.com/lvapeab/nmt-keras/blob/master/score.py) script can be used to obtain the (-log)probabilities of a parallel corpus. Its syntax is the following:
```
python score.py --help
usage: Use several translation models for scoring source--target pairs
       [-h] -ds DATASET [-src SOURCE] [-trg TARGET] [-s SPLITS [SPLITS ...]]
       [-d DEST] [-v] [-c CONFIG] --models MODELS [MODELS ...]
optional arguments:
    -h, --help            show this help message and exit
    -ds DATASET, --dataset DATASET
                            Dataset instance with data
    -src SOURCE, --source SOURCE
                            Text file with source sentences
    -trg TARGET, --target TARGET
                            Text file with target sentences
    -s SPLITS [SPLITS ...], --splits SPLITS [SPLITS ...]
                            Splits to sample. Should be already includedinto the
                            dataset object.
    -d DEST, --dest DEST  File to save scores in
    -v, --verbose         Be verbose
    -c CONFIG, --config CONFIG
                            Config pkl for loading the model configuration. If not
                            specified, hyperparameters are read from config.py
    --models MODELS [MODELS ...]
                            path to the models
  ```
 


## Resources

 * In [examples/documentation/neural_machine_translation.pdf](https://github.com/lvapeab/nmt-keras/blob/master/examples/documentation/neural_machine_translation.pdf) you'll find an overview of an attentional NMT system.

 * In the [examples](https://github.com/lvapeab/nmt-keras/blob/master/examples/) folder you'll find some tutorials for running this library. They are expected to be followed in order:
    
    1) [Dataset set up](https://github.com/lvapeab/nmt-keras/blob/master/examples/1_dataset_tutorial.ipynb): Shows how to invoke and configure a Dataset instance for a translation problem.
    
    2) [Training tutorial](https://github.com/lvapeab/nmt-keras/blob/master/examples/2_training_tutorial.ipynb): Shows how to call a translation model, link it with the dataset object and construct calllbacks for monitorizing the training. 
    
    3) [Decoding tutorial](https://github.com/lvapeab/nmt-keras/blob/master/examples/3_decoding_tutorial.ipynb): Shows how to call a trained translation model and use it to translate new text. 

    4) [NMT model tutorial](https://github.com/lvapeab/nmt-keras/blob/master/examples/4_nmt_model_tutorial.ipynb): Shows how to build a state-of-the-art NMT model with Keras in few (~50) lines. 


## Contact

Álvaro Peris ([web page](http://lvapeab.github.io/)): lvapeab@prhlt.upv.es 

## Acknowledgement

Much of this library has been developed together with [Marc Bolaños](https://github.com/MarcBS) ([web page](http://www.ub.edu/cvub/marcbolanos/)) for other sequence-to-sequence problems. 

To see other projects following the philosophy of NMT-Keras, take a look here:
 
[VIBIKNet for visual question answering](https://github.com/MarcBS/VIBIKNet).

[ABiViRNet for video description](https://github.com/lvapeab/ABiViRNet).

[Sentence SelectioNN for sentence classification and selection](https://github.com/lvapeab/sentence-selectioNN).


## Warning 

NMT-Keras is under development. There are many features still unimplemented. If you find a bug or desire a specific feature, please do not hesitate to contact me.
