# -*- coding: utf-8 -*-
from setuptools import setup

setup(name='nmt_keras',
      version='0.6',
      description='Neural Machine Translation with Keras (Theano and Tensorflow).',
      author='Marc Bolaños - Alvaro Peris',
      author_email='lvapeab@gmail.com',
      url='https://github.com/lvapeab/nmt-keras',
      download_url='https://github.com/lvapeab/nmt-keras/archive/master.zip',
      license='MIT',
      classifiers=[
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: Science/Research',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Topic :: Software Development :: Libraries',
          'Topic :: Software Development :: Libraries :: Python Modules',
          "License :: OSI Approved :: MIT License"
      ],
      install_requires=[
          'coco-caption @ https://github.com/lvapeab/coco-caption/archive/master.zip',
          'cloudpickle',
          'future',
          'keras @ https://github.com/MarcBS/keras/archive/master.zip',
          'keras_applications',
          'keras_preprocessing',
          'h5py',
          'matplotlib',
          'multimodal-keras-wrapper @ https://github.com/lvapeab/multimodal_keras_wrapper/archive/1.0.zip',
          'numpy',
          'scikit-image',
          'scikit-learn',
          'six',
          'tables',
          'numpy',
          'pandas',
          'sacremoses',
          'scipy',
          'tensorflow<2'
      ],
      package_dir={'nmt_keras': '.',
                   'nmt_keras.utils': 'utils',
                   'nmt_keras.data_engine': 'data_engine',
                   'nmt_keras.nmt_keras': 'nmt_keras',
                   'nmt_keras.demo-web': 'demo-web',
                   },
      packages=['nmt_keras',
                'nmt_keras.utils',
                'nmt_keras.data_engine',
                'nmt_keras.nmt_keras',
                'nmt_keras.demo-web'
                ],
      package_data={
          'nmt_keras': ['examples/*']
      }
      )
