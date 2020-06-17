# NMT-Keras

[![Documentation](https://readthedocs.org/projects/nmt-keras/badge/?version=latest)](https://nmt-keras.readthedocs.io) [![Build Status](https://travis-ci.org/lvapeab/nmt-keras.svg)](https://travis-ci.org/lvapeab/nmt-keras) [![Open in colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lvapeab/nmt-keras/blob/master/examples/tutorial.ipynb)  ![Compatibility](https://img.shields.io/badge/Python-3.7-blue.svg) [![license](https://img.shields.io/github/license/mashape/apistatus.svg)](https://github.com/lvapeab/nmt-keras/blob/master/LICENSE)

Neural Machine Translation with Keras.

Library documentation: [nmt-keras.readthedocs.io](https://nmt-keras.readthedocs.io)

<!--<div align="left">-->
  <!--<br><br><img src="https://raw.githubusercontent.com/lvapeab/nmt-keras/master/examples/documentation/attention_nmt_model.png?token=AEf6E5RhGVqGRSmYi87EbtiGZK7lPxrFks5ZAx-KwA%3D%3D"><br><br>-->
<!--</div>-->

<!--<div align="left">-->
  <!--<br><br><img  width="100%" "height:100%" "object-fit: cover" "overflow: hidden" src=""><br><br>-->
<!--</div>-->

## Attentional recurrent neural network NMT model
![alt text](examples/documentation/attention_nmt_model.png "RNN NMT")

## Transformer NMT model
![alt text](examples/documentation/transformer_nmt_model.png "Transformer NMT")
#


## Features (in addition to the full Keras cosmos): .
 * :heavy_exclamation_mark: Multi-GPU training (only for Tensorflow). 
 * [Transformer model](https://arxiv.org/abs/1706.03762).
 * [Tensorboard integration](https://github.com/lvapeab/nmt-keras/blob/master/examples/documentation/tensorboard_integration.md).
 * Online learning and Interactive neural machine translation (INMT). See [the interactive NMT branch](https://github.com/lvapeab/nmt-keras/tree/interactive_NMT).
 * Attention model over the input sequence of annotations.
   - Supporting [Bahdanau (Add)](https://arxiv.org/abs/1409.0473) [Luong (Dot)](https://arxiv.org/abs/1508.04025) attention mechanisms.
   - Also supports double stochastic attention (Eq. 14 from [arXiv:1502.03044](https://arxiv.org/pdf/1502.03044.pdf))
 * Peeked decoder: The previously generated word is an input of the current timestep.
 * Beam search decoding.
 * Ensemble decoding ([sample_ensemble.py](https://github.com/lvapeab/nmt-keras/blob/master/sample_ensemble.py)).
   - Featuring length and source coverage normalization ([reference](https://arxiv.org/abs/1609.08144)).
 * Translation scoring ([score.py](https://github.com/lvapeab/nmt-keras/blob/master/sample_ensemble.py)).
 * Model averaging ([utils/model_average.py](https://github.com/lvapeab/nmt-keras/blob/master/utils/average_models.py)).
 * Support for GRU/LSTM networks:
   - Regular GRU/LSTM units.
   - [Conditional](https://arxiv.org/abs/1703.04357) GRU/LSTM units in the decoder.   
   - Multilayered residual GRU/LSTM networks (and their Conditional version).
 * [Label smoothing](https://arxiv.org/abs/1512.00567).  
 * N-best list generation (as byproduct of the beam search process).
 * Unknown words replacement (see Section 3.3 from [this paper](https://arxiv.org/pdf/1412.2007v2.pdf))
 * Use of pretrained ([Glove](http://nlp.stanford.edu/projects/glove/) or [Word2Vec](https://code.google.com/archive/p/word2vec/)) word embedding vectors.
 * MLPs for initializing the RNN hidden and memory state.
 * [Spearmint](https://github.com/HIPS/Spearmint) [wrapper](https://github.com/lvapeab/nmt-keras/tree/master/meta-optimizers/spearmint) for hyperparameter optimization.
 * [Client-server](https://github.com/lvapeab/nmt-keras/tree/master/demo-web) architecture for web demos:
    - Regular NMT.
    - [Interactive NMT](https://github.com/lvapeab/nmt-keras/tree/interactive_NMT).
    - [Check out the demo!](http://casmacat.prhlt.upv.es/inmt)
    
## Installation

Assuming that you have [pip](https://en.wikipedia.org/wiki/Pip_(package_manager)) installed and updated (>18), run:
  
  ```bash
  git clone https://github.com/lvapeab/nmt-keras
  cd nmt-keras
  pip install -e .
  ```
 
 for installing the library.
 

### Requirements

NMT-Keras requires the following libraries:

 - [Our version of Keras](https://github.com/MarcBS/keras) (Recommended v. 2.0.7 or newer).
 - [Multimodal Keras Wrapper](https://github.com/lvapeab/multimodal_keras_wrapper) (v. 2.0 or newer). ([Documentation](http://marcbs.github.io/staged_keras_wrapper/) and [tutorial](http://marcbs.github.io/multimodal_keras_wrapper/tutorial.html)).


For accelerating the training and decoding on CUDA GPUs, you can optionally install:

 - [CuDNN](https://developer.nvidia.com/cudnn).
 - [CuPy](https://github.com/cupy/cupy).

For evaluating with additional metrics (Meteor, TER, etc), you can use the [Coco-caption evaluation package](https://github.com/lvapeab/coco-caption/tree/master/pycocoevalcap/) and set `METRICS='coco'` in the `config.py` file. This package requires `java` (version 1.8.0 or newer).


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

 * [examples/documentation/nmt-keras_paper.pdf](https://github.com/lvapeab/nmt-keras/blob/master/examples/documentation/nmt-keras_paper.pdf) contains a general overview of the NMT-Keras framework.
 
 * In [examples/documentation/neural_machine_translation.pdf](https://github.com/lvapeab/nmt-keras/blob/master/examples/documentation/neural_machine_translation.pdf) you'll find an overview of an attentional NMT system.

 * In the [examples](https://github.com/lvapeab/nmt-keras/blob/master/examples/) folder you'll find  2 colab notebooks, explaining the basic usage of this library:
 
 * An introduction to a complete NMT experiment: [![Open in colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lvapeab/nmt-keras/blob/master/examples/tutorial.ipynb) 
  * A dissected NMT model: [![Open in colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lvapeab/nmt-keras/blob/master/examples/modeling_tutorial.ipynb) 
 

 * In the [examples/configs](https://github.com/lvapeab/nmt-keras/blob/master/examples/configs) folder you'll find two examples of configs for larger models.

## Citation

If you use this toolkit in your research, please cite:

```
@article{nmt-keras:2018,
 journal = {The Prague Bulletin of Mathematical Linguistics},
 title = {{NMT-Keras: a Very Flexible Toolkit with a Focus on Interactive NMT and Online Learning}},
 author = {\'{A}lvaro Peris and Francisco Casacuberta},
 year = {2018},
 volume = {111},
 pages = {113--124},
 doi = {10.2478/pralin-2018-0010},
 issn = {0032-6585},
 url = {https://ufal.mff.cuni.cz/pbml/111/art-peris-casacuberta.pdf}
}
```


NMT-Keras was used in a number of papers:

* [Online Learning for Effort Reduction in Interactive Neural Machine Translation](https://arxiv.org/abs/1802.03594)
* [Adapting Neural Machine Translation with Parallel Synthetic Data](http://www.statmt.org/wmt17/pdf/WMT14.pdf)
* [Online Learning for Neural Machine Translation Post-editing](https://arxiv.org/pdf/1706.03196.pdf)


### Acknowledgement

Much of this library has been developed together with [Marc Bolaños](https://github.com/MarcBS) ([web page](http://www.ub.edu/cvub/marcbolanos/)) for other sequence-to-sequence problems. 

To see other projects following the same philosophy and style of NMT-Keras, take a look to:

[TMA: Egocentric captioning based on temporally-linked sequences](https://github.com/MarcBS/TMA).

[VIBIKNet: Visual question answering](https://github.com/MarcBS/VIBIKNet).

[ABiViRNet: Video description](https://github.com/lvapeab/ABiViRNet).

[Sentence SelectioNN: Sentence classification and selection](https://github.com/lvapeab/sentence-selectioNN).

[DeepQuest: State-of-the-art models for multi-level Quality Estimation](https://github.com/sheffieldnlp/deepQuest).


### Warning!

The `Theano` backend is not tested anymore, although it should work. There is a [known issue](https://github.com/Theano/Theano/issues/5994) with the `Theano` backend. When running `NMT-Keras`, it will show the following message:

```
[...]
raise theano.gof.InconsistencyError("Trying to reintroduce a removed node")
InconsistencyError: Trying to reintroduce a removed node
```

It is not a critical error, the model keeps working and it is safe to ignore it. However, if you want the message to be gone, use the Theano flag `optimizer_excluding=scanOp_pushout_output`.



## Contact

Álvaro Peris ([web page](http://lvapeab.github.io/)): lvapeab@prhlt.upv.es 

