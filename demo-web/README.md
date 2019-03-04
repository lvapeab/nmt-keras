# Interactive Neural Machine Translation web demo

This is a web demo of an interactive, adaptive neural machine translation system.

![](https://github.com/lvapeab/inmt_demo_web/blob/master/demo-web/images/demo-system.gif)

## Structure

The system follows a client-server architecture. The main files are:

- **sample_server.py** is an HTTP server version of the interactive sampler.
- **sampler.php** and **inmt_sampler.php** query the server to get a translation (given a validated prefix or not).

## How to run a NMT server

In order to provide high response rate and translation speed, it is highly recommended that the server has a GPU available.
By default, the server requires the `InteractiveNMT` branch from the [Multimodal Keras Wrapper](https://github.com/lvapeab/staged_keras_wrapper/tree/Interactive_NMT).

Run ``python sample_server.py --help`` for obtaining information about the options of the server.

We'll start the server in the localhost port 6542. We'll use a dataset instance stored in `datasets/Dataset.pkl` and a 
model and config stored in `trained_models`:
```
python ./sample_server.py --dataset datasets/Dataset.pkl --port=6542  
        --config trained_models/config.pkl --models trained_models/update_15000
```


[Check out the demo!](http://casmacat.prhlt.upv.es/inmt).
