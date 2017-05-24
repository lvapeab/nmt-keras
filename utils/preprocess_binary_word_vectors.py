import numpy as np

# Preprocess pretrained binary vectors
# and stores them in a suitable format (.npy)

# Parameters
ROOT_PATH = '/media/HDD_2TB/DATASETS/'  # Data root path
base_path = ROOT_PATH + 'cnn_polarity/DATA/'  # Binary vectors path
vectors_basename = 'word2vec.'  # Name of the vectors file
language = 'fr'  # Language
dest_file = 'word2vec.' + language  # Destination file

vectors_path = base_path + vectors_basename + language


def word2vec2npy(v_path, base_path_save, dest_filename):
    """
    Preprocess pretrained binary vectors and stores them in a suitable format.
    :param v_path: Path to the binary vectors file.
    :param base_path_save: Path where the formatted vectors will be stored.
    :param dest_filename: Filename of the formatted vectors.
    """
    word_vecs = dict()
    print "Loading vectors from %s" % v_path

    with open(v_path, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        i = 0
        print "Vector length:", layer1_size
        for _ in xrange(vocab_size):
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
                print "Processed %d vectors (%.2f %%)\r" % \
                      (i, 100 * float(i) / vocab_size),

    # Store dict
    print "Saving word vectors in %s" % \
          (base_path_save + '/' + dest_filename + '.npy')
    np.save(base_path_save + '/' + dest_filename + '.npy', word_vecs)
    print


if __name__ == "__main__":
    word2vec2npy(vectors_path, base_path, dest_file)
