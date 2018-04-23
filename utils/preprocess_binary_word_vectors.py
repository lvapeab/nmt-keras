# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import argparse
from os.path import basename, dirname


# Preprocess pretrained binary vectors
# and stores them in a suitable format (.npy)

def word2vec2npy(v_path, base_path_save, dest_filename):
    """
    Preprocess pretrained binary vectors and stores them in a suitable format.
    :param v_path: Path to the binary vectors file.
    :param base_path_save: Path where the formatted vectors will be stored.
    :param dest_filename: Filename of the formatted vectors.
    """
    word_vecs = dict()
    print ("Loading vectors from %s" % v_path)
    with open(v_path, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        i = 0
        print ("Vector length:", layer1_size)
        for _ in range(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            word_vecs[word] = np.fromstring(f.read(binary_len),
                                            dtype='float32')
            i += 1
            if i % 1000 == 0:
                print ("Processed %d vectors (%.2f %%)\r" % (i, 100 * float(i) / vocab_size),)

    # Store dict
    print ("")
    print ("Saving word vectors in %s" % (base_path_save + '/' + dest_filename + '.npy'))
    np.save(base_path_save + '/' + dest_filename + '.npy', word_vecs)
    print("")


def parse_args():
    parser = argparse.ArgumentParser("Preprocess pre-trained word embeddings.")
    parser.add_argument("-v", "--vectors", required=True, help="Pre-trained word embeddings file.",
                        default="GoogleNews-vectors-negative300.bin")
    parser.add_argument("-d", "--destination", required=True, help="Destination file.", default='word2vec.en')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dest_file = basename(args.destination)
    base_path = dirname(args.destination)

    word2vec2npy(args.vectors, base_path, dest_file)
