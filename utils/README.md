# NMT-Keras utils

In this directory, you'll find some utils for handling NMT systems. 
The main scripts are the following:

* [build_mapping_file.sh](https://github.com/lvapeab/nmt-keras/blob/master/utils/build_mapping_file.sh): Given a parallel corpus, estimates a mapping (through a stochastic dictionary) of source-target words. Used for replace unknown words heuristics 1 and 2.
* [evaluate_from_file.py](https://github.com/lvapeab/nmt-keras/blob/master/utils/evaluate_from_file.py): Applies the selected metrics to hypotheses/references files.
* [preprocess_binary_word_vectors.py](https://github.com/lvapeab/nmt-keras/blob/master/utils/preprocess_binary_word_vectors.py): Formats word2vec (or GloVe) word embeddings given in a binary format. You should change the paths to yours adequately.
* [preprocess_text_word_vectors.py](https://github.com/lvapeab/nmt-keras/blob/master/utils/preprocess_text_word_vectors.py): Formats word2vec (or GloVe) word embeddings given in text format. You should change the paths to yours adequately.
* [vocabulary_size.sh](https://github.com/lvapeab/nmt-keras/blob/master/utils/vocabulary_size.sh): Computes the size of the vocabulary of the input files.

