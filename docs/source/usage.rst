#####
Usage
#####

********
Training
********

1) Set a training configuration in the config.py_ script. Each parameter is commented. See the `documentation file`_ for further info about each specific hyperparameter. You can also specify the parameters when calling the `main.py`_ script following the syntax `Key=Value`

2) Train!::

    python main.py

********
Decoding
********
Once we have our model trained, we can translate new text using the `sample_ensemble.py`_ script. Please refer to the `ensembling tutorial`_ for more details about this script.
In short, if we want to use the models from the first three epochs to translate the `examples/EuTrans/test.en` file, just run::

    python sample_ensemble.py --models trained_models/tutorial_model/epoch_1 \
                                       trained_models/tutorial_model/epoch_2 \
                              --dataset datasets/Dataset_tutorial_dataset.pkl \
                              --text examples/EuTrans/test.en

*******
Scoring
*******

The `score.py`_ script can be used to obtain the (-log)probabilities of a parallel corpus. Its syntax is the following::

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
                            Splits to sample. Should be already included into the
                            dataset object.
        -d DEST, --dest DEST  File to save scores in
        -v, --verbose         Be verbose
        -c CONFIG, --config CONFIG
                            Config pkl for loading the model configuration. If not
                            specified, hyperparameters are read from config.py
        --models MODELS [MODELS ...]
                            path to the models




.. _documentation file: https://github.com/lvapeab/nmt-keras/blob/master/examples/documentation/config.md
.. _config.py: https://github.com/lvapeab/nmt-keras/blob/master/config.py
.. _main.py: https://github.com/lvapeab/nmt-keras/blob/master/main.py
.. _sample_ensemble.py: https://github.com/lvapeab/nmt-keras/blob/master/sample_ensemble.py
.. _ensembling tutorial: https://github.com/lvapeab/nmt-keras/blob/master/examples/documentation/ensembling_tutorial.md
.. _score.py: https://github.com/lvapeab/nmt-keras/blob/master/score.py

