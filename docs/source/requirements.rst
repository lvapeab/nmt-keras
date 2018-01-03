############
Installation
############
Assuming that you have pip_ installed, run::

    git clone https://github.com/lvapeab/nmt-keras
    cd nmt-keras
    pip install -r requirements.txt

for obtaining the required packages for running this library.

Nevertheless, it is highly recommended to install and configure Theano_ or Tensorflow_ with the GPU and speed optimizations enabled.

Requirements
************
 - Our version of Keras_.
 - `Multimodal Keras Wrapper`_. See the documentation_ and tutorial_.
 - Coco-caption_ evaluation package (Only required to perform evaluation).

.. _Keras: https://github.com/MarcBS/keras
.. _Multimodal Keras Wrapper: https://github.com/lvapeab/multimodal_keras_wrapper
.. _documentation: http://marcbs.github.io/staged_keras_wrapper/
.. _tutorial: http://marcbs.github.io/multimodal_keras_wrapper/tutorial.html
.. _Coco-caption: https://github.com/lvapeab/coco-caption
.. _pip: https://en.wikipedia.org/wiki/Pip_(package_manager)
.. _Theano: http://theano.readthedocs.io/en/latest/install.html#install
.. _Tensorflow: https://www.tensorflow.org/install/