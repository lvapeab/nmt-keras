import numpy as np
import argparse
from os.path import basename, dirname

# Preprocess pretrained text vectors
# and stores them in a suitable format (.npy)

def txtvec2npy(v_path, base_path_save, dest_filename):
    """
    Preprocess pretrained text vectors and stores them in a suitable format
    :param v_path: Path to the text vectors file.
    :param base_path_save: Path where the formatted vectors will be stored.
    :param dest_filename: Filename of the formatted vectors.
    """
    word_vecs = dict()
    print "Loading vectors from %s" % v_path
    glove_vectors = [x[:-1] for x in open(v_path).readlines()]
    n_vecs = len(glove_vectors)
    print "Found %d vectors in %s" % (n_vecs, v_path)
    i = 0
    for vector in glove_vectors:
        v = vector.split()
        word = v[0]
        vec = np.asarray(v[1:], dtype='float32')
        word_vecs[word] = vec
        i += 1
        if i % 1000 == 0:
            print "Processed %d vectors (%.2f %%)\r" % \
                  (i, 100 * float(i) / n_vecs),

    print
    # Store dict
    print "Saving word vectors in %s" %\
          (base_path_save + '/' + dest_filename + '.npy')
    np.save(base_path_save + '/' + dest_filename + '.npy', word_vecs)
    print

def parse_args():
    parser = argparse.ArgumentParser("Preprocess pre-trained word embeddings.")
    parser.add_argument("-v", "--vectors", required=True, help="Pre-trained word embeddings file.",
                        default="GoogleNews-vectors-negative300.txt")
    parser.add_argument("-d", "--destination", required=True, help="Destination file.", default='word2vec.en')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    dest_file = basename(args.destination)
    base_path = dirname(args.destination)
    txtvec2npy(args.vectors, base_path, dest_file)
