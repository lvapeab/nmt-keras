# Interactive Neural Machine Translation web demo

This is a web demo of an interactive, adaptive neural machine translation system.

![](https://github.com/lvapeab/inmt_demo_web/blob/master/demo-web/images/demo-system.gif)

## Structure

The system follows a client-server architecture. Take a look [here](https://github.com/lvapeab/inmt_demo_web) for a client website.

## How to run a NMT server

In order to provide high response rate and translation speed, it is recommended that the server has a GPU available.
By default, the server requires the `InteractiveNMT` branch from the [Multimodal Keras Wrapper](https://github.com/lvapeab/staged_keras_wrapper/tree/Interactive_NMT).

Run ``python3 sample_server.py --help`` for obtaining information about the options of the server.

We'll start the server in the localhost port 8001 from the address 127.0.0.1. We'll use a dataset instance stored in `datasets/Dataset.pkl` and a 
model and config stored in `trained_models`:
```
python3 ./sample_server.py --dataset datasets/Dataset.pkl --address 127.0.0.1 --port=8001  
        --config trained_models/config.pkl --models trained_models/update_15000
```


[Check out the demo!](http://casmacat.prhlt.upv.es/interactive-seq2seq/).
