import numpy as np

# Preprocess pretrained text vectors
# and stores them in a suitable format (.npy)

# Parameters
ROOT_PATH = '/media/HDD_2TB/DATASETS/'  # Data root path
base_path = ROOT_PATH + 'cnn_polarity/DATA/'  # Binary vectors path
vectors_basename = 'word2vec.'  # Name of the vectors file
language = 'fr'  # Language
dest_file = 'word2vec.' + language  # Destination file

vectors_path = base_path + vectors_basename + language


def txtvec2npy(v_path, base_path_save, dest_filename):
    """
    Preprocess pretrained text vectors and stores them in a suitable format
    :param v_path: Path to the text vectors file.
    :param base_path_save: Path where the formatted vectors will be stored.
    :param dest_filename: Filename of the formatted vectors.
    """
    vecs_dict = dict()
    print "Loading vectors from %s" % v_path

    glove_vectors = [x[:-1] for x in open(v_path).readlines()]
    n_vecs = len(glove_vectors)
    print "Found %d vectors in %s" % (n_vecs, v_path)
    i = 0
    for vector in glove_vectors:
        v = vector.split()
        word = v[0]
        vec = np.asarray(v[1:], dtype='float32')
        vecs_dict[word] = vec
        i += 1
        if i % 1000 == 0:
            print "Processed %d vectors (%.2f %%)\r" % \
                  (i, 100 * float(i) / n_vecs),

    print
    # Store dict
    print "Saving word vectors in %s" %\
          (base_path_save + '/' + dest_filename + '.npy')
    np.save(base_path_save + '/' + dest_filename + '.npy', vecs_dict)
    print


if __name__ == "__main__":
    txtvec2npy(vectors_path, base_path, dest_file)
