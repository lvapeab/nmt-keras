from __future__ import print_function

import argparse
import logging
import numpy as np
import os
from keras_wrapper.extra.read_write import file2list, numpy2file
from nltk import ngrams

logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser("Computes a the centroid of the average continuous representation from a file,"
                                     "from pretrained word embeddings.")

    parser.add_argument("-d", "--dest",
                        default='./ngrams',
                        required=False,
                        help="Path to store the centroid. If not specified, the model is saved in './centroid.pkl'.")
    parser.add_argument("-v", "--verbose", required=False, default=0, type=int, help="Verbosity level")
    parser.add_argument("-s", "--sentence-mode", default='average', type=str, help="Mode of computing a sentence "
                                                                                   "representation from the representations "
                                                                                   "of the words of it. "
                                                                                   "One of: 'average' and 'sum'.")
    parser.add_argument("-w", "--word-embeddings", required=True, type=str, help="Path to the processed word"
                                                                                 "embeddings used for computing the "
                                                                                 "representation.")
    parser.add_argument("-c", "--corpus", required=True, help="Corpus on which extract the n-grams.")

    return parser.parse_args()


def compute_centroid(args):
    corpus = file2list(args.corpus)
    len_corpus = len(corpus)
    print("Loading word vectors from %s " % (args.word_embeddings))
    word_vectors = np.load(os.path.join(args.word_embeddings)).item()
    word_dim = word_vectors[word_vectors.keys()[0]].shape[0]
    centroid = np.zeros(word_dim)
    print ("Done.")
    for i, sentence in enumerate(corpus):
        if i % 1000 == 0 and args.verbose == 1:
            print ("Processed %d sentences (%.2f %%)\r" % (i, 100 * float(i) / len_corpus))
        sentence_representation= np.zeros(word_dim)
        for word in sentence.split():
            sentence_representation += word_vectors.get(word, word_vectors.get('unk', np.zeros(word_dim)))
        if args.sentence_mode == 'average':
            sentence_representation /= len(sentence.split())
        centroid += sentence_representation
    centroid /= len_corpus

    print ("Finished. Storing n-gram counts in %s " % args.dest)
    numpy2file(args.dest, centroid)

if __name__ == "__main__":
    args = parse_args()
    assert args.sentence_mode in ['average', 'sum'], 'Unknown sentence-mode: "%s"' % args.sentence_mode
    compute_centroid(args)

