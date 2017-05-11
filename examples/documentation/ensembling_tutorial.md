This document describes the usage of the [sample_ensemble](https://github.com/lvapeab/nmt-keras/blob/master/sample_ensemble.py) script.

If we launch the script with the `--help` flag, the different options are shown:
```bash
 python sample_ensemble.py --help
 usage: Apply several translation models for making predictions
       [-h] -ds DATASET [-t TEXT] [-s SPLITS [SPLITS ...]] [-d DEST]
       [--not-eval] [-e EVAL_OUTPUT] [-v] [-c CONFIG] --models MODELS
       [MODELS ...]

  optional arguments:
    -h, --help            show this help message and exit
    -ds DATASET, --dataset DATASET
                          Dataset instance with data
    -t TEXT, --text TEXT  Text file with source sentences
    -s SPLITS [SPLITS ...], --splits SPLITS [SPLITS ...]
                          Splits to sample. Should be already includedinto the
                          dataset object.
    -d DEST, --dest DEST  File to save translations in
    -v, --verbose         Be verbose
    -c CONFIG, --config CONFIG
                          Config pkl for loading the model configuration. If not
                          specified, hyperparameters are read from config.py
    --n-best              Write n-best list (n = beam size)                       
    --models MODELS [MODELS ...]
```

The main arguments are the following: 
* ``--dataset DATASET``: Path to the dataset instance used for training the model. **REQUIRED** since it establishes several hyperparameters, index2word mappings, etc.
* ``--text TEXT``: Path to a text file with source sentences. If this is specified, the model will translate only the sources sentences from this file.
* ``--splits SPLITS [SPLITS ...]``: List of splits to sample (e.g.: ``val test``). Should be already included into the dataset object. If the ``--text`` option is activated, this is ignored.
* ``--dest DEST``: Path to a file to save translations in. If not specified, the translations won't be stored.
* ``--config CONFIG``: Config pkl for loading the model configuration. If not specified, hyperparameters are read from ``config.py``
* ``--models MODELS [MODELS ...]``: List of models to load. **REQUIRED**. Here, we only need to specify the prefix of each model. For instance, if we want to sample from the models from epochs 1, 2 and 3 from models stored in the ``trained_models`` folder, this option should be: ``--models trained_models/epoch_1 trained_models/epoch_2 trained_models/epoch_3``.
