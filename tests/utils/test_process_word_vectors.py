import inspect
import os
import unittest
import numpy as np
from subprocess import call

from utils.preprocess_text_word_vectors import txtvec2npy

class TestEvaluateFromFile(unittest.TestCase):
    def test_text_word2vec2npy(self):
        # check whether files are present in folder
        vectors_name = 'wiki.fiu_vro.vec'
        path = os.path.dirname(inspect.getfile(inspect.currentframe()))
        if not os.path.exists(path + '/' + vectors_name):
            call(["wget https://s3-us-west-1.amazonaws.com/fasttext-vectors/" + vectors_name + " -O "
                  + path + "/" + vectors_name],
                 shell=True)
        txtvec2npy(path + '/' + vectors_name, './', vectors_name[:-4])
        vectors = np.load('./' + vectors_name[:-4] + '.npy').item()
        self.assertEqual(len(vectors.keys()), 8770)
        self.assertEqual(vectors['kihlkunnan'].shape[0], 300)

if __name__ == '__main__':
    unittest.main()
