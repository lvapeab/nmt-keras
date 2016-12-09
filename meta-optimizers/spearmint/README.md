Package for performing hyperparameter optimization with [Spearmint] (https://github.com/HIPS/Spearmint).

Requirements:  Those specified in the [Spearmint] (https://github.com/HIPS/Spearmint) package:

* [NumPy](http://www.numpy.org/)
* [scikit learn](http://scikit-learn.org/stable/index.html)
* [pymongo](https://api.mongodb.org/python/current)
* [MongoDB](https://www.mongodb.org)

Installation: 

* Install [Spearmint] (https://github.com/HIPS/Spearmint/blob/master/README.md)

Usage:


 * Get NMT-Keras directory:
 
 ```bash
    cd nmt-keras
    nmt_keras_path=`pwd`
 ```
 
 * Set your experimental settings (see `${nmt_keras_path}/spearmint/config.json` for an example)
  
 * Create directory for storing the database:
 
 ```bash
 mkdir ${nmt_keras_path}/spearmint/db
 ```
 
 * Start the Mongo database:
 
 ```bash
 mongod --fork --logpath ${nmt_keras_path}/spearmint/db/log --dbpath ${nmt_keras_path}/spearmint/db
 ```
 
  * Remove eventual instances of previous experiments
  
 ```bash
  ${spearmint_path}/spearmint/cleanup.sh ${nmt_keras_path}/spearmint/
 ```
 
 * Lauch Spearmint! Assuming that it is installed under `${spearmint_path}`:
 
    ```bash
    nohup python ${spearmint_path}/spearmint/main.py  ${nmt_keras_path}/spearmint --config=${nmt_keras_path}/spearmint/config.json > logs/spearmint.log 2> logs/spearmint.err &
    ```
    
 * The results will appear at `${nmt_keras_path}/spearmint/output` 