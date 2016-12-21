import numpy as np

# Parameters
ROOT_PATH = '/media/HDD_2TB/DATASETS/'
base_path = ROOT_PATH + 'cnn_polarity/DATA/'
language = 'fr'
vectors_path = base_path + 'word2vec_bin.' + language
dest_file = 'word2vec.' + language


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
