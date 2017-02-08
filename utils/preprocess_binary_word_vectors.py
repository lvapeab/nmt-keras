import numpy as np

"""
Preprocess pretrained binary vectors and stores them in a suitable format (.npy)
"""

# Parameters
<<<<<<< HEAD:utils/preprocess_binary_vectors.py
ROOT_PATH = '/media/HDD_2TB/antonio/'                   # Data root path
base_path = ROOT_PATH + 'pretrainedVectors/source'             # Binary vectors path
vectors_basename = 'word2vecb.'                     # Name of the vectors file
language = 'en'                                          # Language
dest_file = '../word2vec.' + language                       # Destination file
=======
ROOT_PATH = '/media/HDD_2TB/DATASETS/'  # Data root path
base_path = ROOT_PATH + 'cnn_polarity/DATA/'  # Binary vectors path
vectors_basename = 'word2vec.'  # Name of the vectors file
language = 'fr'  # Language
dest_file = 'word2vec.' + language  # Destination file
>>>>>>> 7789462dea59a80b04da54a469d27b9fa3ea4909:utils/preprocess_binary_word_vectors.py

vectors_path = base_path + vectors_basename + language


def word2vec2npy(v_path, base_path_save, dest_filename):
    word_vecs = dict()
    print "Loading vectors from %s" % v_path

    with open(v_path, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        i = 0
        print "Vector length:", layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
            i += 1
            if i % 1000 == 0:
                print "Processed %d vectors (%.2f %%)\r" % (i, 100 * float(i) / vocab_size),

    # Store dict
    print "Saving word vectors in %s" % (base_path_save + '/' + dest_filename + '.npy')
    np.save(base_path_save + '/' + dest_filename + '.npy', word_vecs)
    print


if __name__ == "__main__":
    word2vec2npy(vectors_path, base_path, dest_file)
