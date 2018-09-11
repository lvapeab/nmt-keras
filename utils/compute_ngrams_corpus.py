from __future__ import print_function

import argparse
import logging

from keras_wrapper.extra.read_write import file2list, dict2pkl
from nltk import ngrams

logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser("Computes all the n-grams from a file.")

    parser.add_argument("-d", "--dest",
                        default='./ngrams',
                        required=False,
                        help="Path to store the N-grams file. If not specified, the model is saved in './ngrams.pkl'.")
    parser.add_argument("-v", "--verbose", required=False, default=0, type=int, help="Verbosity level")
    parser.add_argument("-o", "--order", default=4, type=int, help="Maximum order of n-grams to consider.")
    parser.add_argument("-c", "--corpus", required=True, help="Corpus on which extract the n-grams.")
    return parser.parse_args()


def extract_n_grams(args):
    n_gram_counts = dict()
    corpus = file2list(args.corpus)
    len_corpus = len(corpus)
    for i, sentence in enumerate(corpus):
        if i % 1000 == 0 and args.verbose == 1:
            print ("Processed %d sentences (%.2f %%)\r" % (i, 100 * float(i) / len_corpus))
        for order in range(1, args.order + 1):
            sentence_ngrams = ngrams(sentence.split(), order)
            for ngram in sentence_ngrams:
                ngram_string = u' '.join(ngram)
                if n_gram_counts.get(ngram_string) is not None:
                    n_gram_counts[ngram_string] = n_gram_counts[ngram_string] + 1
                else:
                    n_gram_counts[ngram_string] = 1
    print ("Finished. Storing n-gram counts in %s " % args.dest)
    dict2pkl(n_gram_counts, args.dest)

if __name__ == "__main__":
    args = parse_args()
    extract_n_grams(args)
