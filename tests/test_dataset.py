import unittest
from config import load_parameters
from data_engine.prepare_data import build_dataset
from keras_wrapper.dataset import Dataset, loadDataset


class TestDataset(unittest.TestCase):

    def test_build_datset(self):
        params = load_parameters()
        params['REBUILD_DATASET'] = True
        params['DATASET_STORE_PATH'] = './'
        ds = build_dataset(params)
        self.assertIsInstance(ds, Dataset)

    def test_load_dataset(self):
        params = load_parameters()
        ds = loadDataset('./Dataset_' +
                         params['DATASET_NAME'] +
                         '_' + params['SRC_LAN'] +
                         params['TRG_LAN'] + '.pkl')
        self.assertIsInstance(ds, Dataset)
        self.assertIsInstance(ds.vocabulary, dict)
        self.assertGreaterEqual(ds.vocabulary.keys(), 3)
        [self.assertEqual(len(ds.vocabulary[voc].keys()), 2)
         for voc in ds.vocabulary]

if __name__ == '__main__':
    unittest.main()
