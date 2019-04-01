import inspect
import os
import pytest
import numpy as np
from subprocess import call
from utils.preprocess_text_word_vectors import txtvec2npy


def test_text_word2vec2npy():
    # check whether files are present in folder
    vectors_name = 'wiki.fiu_vro.vec'
    path = os.path.dirname(inspect.getfile(inspect.currentframe()))
    if not os.path.exists(path + '/' + vectors_name):
        call(["wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/" + vectors_name + " -O " +
              path + "/" + vectors_name],
             shell=True)
    txtvec2npy(path + '/' + vectors_name, './', vectors_name[:-4])
    vectors = np.load('./' + vectors_name[:-4] + '.npy').item()

    assert len(list(vectors)) == 8769
    assert vectors['kihlkunnan'].shape[0] == 300


if __name__ == '__main__':
    pytest.main([__file__])
