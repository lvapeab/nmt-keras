import unittest
from config import load_parameters
from data_engine.prepare_data import build_dataset
from keras_wrapper.dataset import Dataset, loadDataset


class TestDataset(unittest.TestCase):

    def test_build_datset(self):
        params = load_parameters()
        params['REBUILD_DATASET'] = True
        ds = build_dataset(params)
        self.assertIsInstance(ds, Dataset)

    def test_load_dataset(self):
        ds = loadDataset('datasets/Dataset_EuTrans_esen.pkl')
        self.assertIsInstance(ds, Dataset)
        self.assertIsInstance(ds.vocabulary, dict)
        self.assertGreaterEqual(ds.vocabulary.keys(), 3)
        [self.assertEqual(len(ds.vocabulary[voc].keys()), 2) for voc in ds.vocabulary]

if __name__ == '__main__':
    unittest.main()