# Interactive Neural Machine Translation web demo

This is a web demo of an interactive, adaptive neural machine translation system.

![](https://github.com/lvapeab/nmt-keras/blob/master/demo-web/images/demo-system.gif)

## Structure

The system follows a client-server architecture. The main files are:

- **index.html** and **document_translation.html**: HTML webpage. 
- **sample_server.py** is an HTTP server version of the interactive sampler.
- **sampler.php** and **inmt_sampler.php** query the server to get a translation (given a validated prefix or not).
- **load_file.php** loads a plain-text file.
- The `images` and `assets` directories contain some visual resources. 

##How to run a demo server

In order to provide high response rate and translation speed, it is highly recommended that the server has a GPU available.
By default, the server requires the `InteractiveNMT` branch from the [Multimodal Keras Wrapper](https://github.com/lvapeab/staged_keras_wrapper/tree/Interactive_NMT).

Run ``python sample_server.py --help`` for obtaining information about the options of the server.

We'll start the server in the localhost port 8888. We'll use a dataset instance stored in `datasets/Dataset.pkl` and a 
model and config stored in `trained_models`:
```
python ./sample_server.py --dataset datasets/Dataset.pkl --port=8888  
        --config trained_models/config.pkl --models trained_models/update_15000
```

Finally, we need to run our php server. For running it in localhost, we just execute:
```
php -S localhost:8000
```

Finally, we should make `sampler.php` match our `sample_server.py` port. In `sampler.php` and `inmt_sampler.php`
the lines 
```
$url = 'http://localhost:8888/?source='.urlencode($source);
```
and
```
$url = 'http://localhost:8888/?source='.urlencode($source).'&prefix='.urlencode($prefix);
```
should point to your php server.
