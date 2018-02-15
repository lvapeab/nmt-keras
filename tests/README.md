# NMT-Keras test suite 

Unit tests of NMT-Keras.

## Requirements:

```
pytest
pytest-pep8
pytest-cache
pytest-cov
pytest-forked
pytest-xdist
coverage
```



## Running tests:

Run tests with `pytest` from the main NMT-Keras directory. For running all tests (with coverage): 

``
py.test  tests/ --cov-config .coveragerc   --cov=. tests/
``

For running a specific test (e.g.):

``
py.test tests/test_load_params.py
``

For running the pep8 formatting test:
``
PYTHONPATH=$PWD:$PYTHONPATH pytest --pep8 -m pep8 -n0;
``

## TODO

 - Write more tests
 - Write more documentation
 



