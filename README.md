# NMT-Keras

[![Documentation](https://readthedocs.org/projects/nmt-keras/badge/?version=latest)](https://nmt-keras.readthedocs.io) [![Build Status](https://travis-ci.org/lvapeab/nmt-keras.svg?branch=master)](https://travis-ci.org/lvapeab/nmt-keras) [![Codacy Badge](https://api.codacy.com/project/badge/Grade/1239923bcbba438b97b374ae8dc435be)](https://www.codacy.com/app/lvapeab/nmt-keras?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=lvapeab/nmt-keras&amp;utm_campaign=Badge_Grade) [![Requirements Status](https://requires.io/github/lvapeab/nmt-keras/requirements.svg?branch=master)](https://requires.io/github/lvapeab/nmt-keras/requirements/?branch=master) [![license](https://img.shields.io/github/license/mashape/apistatus.svg)]()

Neural Machine Translation with Keras (+ Theano backend).

Library documentation: [nmt-keras.readthedocs.io](https://nmt-keras.readthedocs.io)

<div align="left">
  <br><br><img src="https://raw.githubusercontent.com/lvapeab/nmt-keras/master/examples/documentation/attention_nmt_model.png?token=AEf6E5RhGVqGRSmYi87EbtiGZK7lPxrFks5ZAx-KwA%3D%3D"><br><br>
</div>

# WARNING!!

You are now in the `InteractiveNMT` branch of NMT-Keras. This branch is designed to implement the interactive protocols described in the paper [Interactive Neural Machine Translation](http://www.sciencedirect.com/science/article/pii/S0885230816301000): 

If you use this repository for any purpose, please the aforementioned paper:
 
```
Interactive Neural Machine Translation
Álvaro Peris, Miguel Domingo and Francisco Casacuberta.
In Computer Speech & Language. In press, 2016.
```

The interactive simulation is implemented at the `interactive_nmt_simulation.py` script. 

In addition, this branch also supports online learning. Refer to the `main.py` for more information.

## Requirements
    
This branch requires—in addition to the regular NMT-Keras requirements—the `InteractiveNMT` branch from [Staged Keras Wrapper](https://github.com/lvapeab/staged_keras_wrapper/tree/Interactive_NMT).


## Features (in addition to the full Keras cosmos): 

 * Attention model over the input sequence of annotations.
 * Peeked decoder: The previously generated word is an input of the current timestep.
 * Beam search decoding.
 * Ensemble decoding ([sample_ensemble.py](https://github.com/lvapeab/nmt-keras/blob/master/sample_ensemble.py)).
   - Featuring length and source coverage normalization ([reference](https://arxiv.org/abs/1609.08144)).
 * Translation scoring ([score.py](https://github.com/lvapeab/nmt-keras/blob/master/sample_ensemble.py)).
 * Support for GRU/LSTM networks.
 * Multilayered residual GRU/LSTM networks.
 * N-best list generation (as byproduct of the beam search process).
 * Unknown words replacement (see Section 3.3 from [this paper](https://arxiv.org/pdf/1412.2007v2.pdf))
 * Use of pretrained ([Glove](http://nlp.stanford.edu/projects/glove/) or [Word2Vec](https://code.google.com/archive/p/word2vec/)) word embedding vectors.
 * MLPs for initializing the RNN hidden and memory state.
 * [Spearmint](https://github.com/HIPS/Spearmint) [wrapper](https://github.com/lvapeab/nmt-keras/tree/master/meta-optimizers/spearmint) for hyperparameter optimization

## Installation

Assuming that you have [pip](https://en.wikipedia.org/wiki/Pip_(package_manager)) installed, run:
  
  ```bash
  git clone https://github.com/lvapeab/nmt-keras
  cd nmt-keras
  pip install -r requirements.txt
  ```
 
 for obtaining the required packages for running this library.
 

### Requirements

NMT-Keras requires the following libraries:

 - [Our version of Keras](https://github.com/MarcBS/keras) (Recommended v. 1.2.3 or newer, as it solves some issues)
 - [Multimodal Keras Wrapper](https://github.com/lvapeab/multimodal_keras_wrapper) (v. 0.7 or newer) ([Documentation](http://marcbs.github.io/staged_keras_wrapper/) and [tutorial](http://marcbs.github.io/multimodal_keras_wrapper/tutorial.html))
 - [Coco-caption evaluation package](https://github.com/lvapeab/coco-caption/tree/master/pycocoevalcap/) (Only required to perform evaluation)


## Usage

### Training
 1) Set a training configuration in the `config.py` script. Each parameter is commented. See the [documentation file](https://github.com/lvapeab/nmt-keras/blob/master/examples/documentation/config.md) for further info about each specific hyperparameter.
 You can also specify the parameters when calling the `main.py` script following the syntax `Key=Value`

 2) Train!:

  ``
 python main.py
 ``


### Decoding
 Once we have our model trained, we can translate new text using the [sample_ensemble.py](https://github.com/lvapeab/nmt-keras/blob/master/sample_ensemble.py) script. Please refer to the [ensembling_tutorial](https://github.com/lvapeab/nmt-keras/blob/master/examples/documentation/ensembling_tutorial.md) for more details about this script. 
In short, if we want to use the models from the first three epochs to translate the `examples/EuTrans/test.en` file, just run:
 ```bash
  python sample_ensemble.py 
              --models trained_models/tutorial_model/epoch_1 \ 
                       trained_models/tutorial_model/epoch_2 \
              --dataset datasets/Dataset_tutorial_dataset.pkl \
              --text examples/EuTrans/test.en
  ```
 
 
 ### Scoring
 
 The [score.py](https://github.com/lvapeab/nmt-keras/blob/master/score.py) script can be used to obtain the (-log)probabilities of a parallel corpus. Its syntax is the following:
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
  
  
 ### Advanced features
 Other features such as online learning or interactive NMT protocols are implemented in the [interactiveNMT](https://github.com/lvapeab/nmt-keras/tree/interactive_NMT) branch.


## Resources

 * In [examples/documentation/neural_machine_translation.pdf](https://github.com/lvapeab/nmt-keras/blob/master/examples/documentation/neural_machine_translation.pdf) you'll find an overview of an attentional NMT system.

 * In the [examples](https://github.com/lvapeab/nmt-keras/blob/master/examples/) folder you'll find some tutorials for running this library. They are expected to be followed in order:
    
    1) [Dataset set up](https://github.com/lvapeab/nmt-keras/blob/master/examples/1_dataset_tutorial.ipynb): Shows how to invoke and configure a Dataset instance for a translation problem.
    
    2) [Training tutorial](https://github.com/lvapeab/nmt-keras/blob/master/examples/2_training_tutorial.ipynb): Shows how to call a translation model, link it with the dataset object and construct calllbacks for monitorizing the training. 
    
    3) [Decoding tutorial](https://github.com/lvapeab/nmt-keras/blob/master/examples/3_decoding_tutorial.ipynb): Shows how to call a trained translation model and use it to translate new text. 

    4) [NMT model tutorial](https://github.com/lvapeab/nmt-keras/blob/master/examples/4_nmt_model_tutorial.ipynb): Shows how to build a state-of-the-art NMT model with Keras in few (~50) lines. 


## Acknowledgement

Much of this library has been developed together with [Marc Bolaños](https://github.com/MarcBS) ([web page](http://www.ub.edu/cvub/marcbolanos/)) for other sequence-to-sequence problems. 

To see other projects following the philosophy of NMT-Keras, take a look here:

[TMA for egocentric captioning based on temporally-linked sequences](https://github.com/MarcBS/TMA).

[VIBIKNet for visual question answering](https://github.com/MarcBS/VIBIKNet).

[ABiViRNet for video description](https://github.com/lvapeab/ABiViRNet).

[Sentence SelectioNN for sentence classification and selection](https://github.com/lvapeab/sentence-selectioNN).


## Contact

Álvaro Peris ([web page](http://lvapeab.github.io/)): lvapeab@prhlt.upv.es 

