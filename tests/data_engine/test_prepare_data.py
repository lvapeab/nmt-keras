import copy
import unittest

from config import load_parameters
from data_engine.prepare_data import build_dataset, update_dataset_from_file
from keras_wrapper.dataset import Dataset, loadDataset


class TestDataset(unittest.TestCase):
    def test_build_datset(self):
        params = load_parameters()
        params['REBUILD_DATASET'] = True
        params['DATASET_STORE_PATH'] = './'
        ds = build_dataset(params)
        self.assertIsInstance(ds, Dataset)
        len_splits = [('train', 9900), ('val', 100), ('test', 2996)]
        for split, len_split in len_splits:
            self.assertEqual(eval('ds.len_' + split), len_split)
            self.assertTrue(eval('all(ds.loaded_' + split + ')'))
            self.assertEqual(len(eval('ds.X_' + split + str([params['INPUTS_IDS_DATASET'][0]]))), len_split)
            self.assertEqual(len(eval('ds.Y_' + split + str([params['OUTPUTS_IDS_DATASET'][0]]))), len_split)

    def test_load_dataset(self):
        params = load_parameters()
        ds = loadDataset('./Dataset_' + params['DATASET_NAME'] + '_' + params['SRC_LAN'] + params['TRG_LAN'] + '.pkl')
        self.assertIsInstance(ds, Dataset)
        self.assertIsInstance(ds.vocabulary, dict)
        self.assertGreaterEqual(ds.vocabulary.keys(), 3)
        for voc in ds.vocabulary:
            self.assertEqual(len(ds.vocabulary[voc].keys()), 2)

    def test_update_dataset_from_file(self):
        params = load_parameters()
        params['REBUILD_DATASET'] = True
        params['DATASET_STORE_PATH'] = './'
        ds = build_dataset(params)
        self.assertIsInstance(ds, Dataset)
        for splits in [[], ['val']]:
            for output_text_filename in [None,
                                         params['DATA_ROOT_PATH'] + params['TEXT_FILES']['test'] + params['TRG_LAN']]:
                for remove_outputs in [True, False]:
                    for compute_state_below in [True, False]:
                        for recompute_references in [True, False]:
                            ds2 = update_dataset_from_file(copy.deepcopy(ds),
                                                           params['DATA_ROOT_PATH'] + params['TEXT_FILES']['test'] +
                                                           params['SRC_LAN'],
                                                           params,
                                                           splits=splits,
                                                           output_text_filename=output_text_filename,
                                                           remove_outputs=remove_outputs,
                                                           compute_state_below=compute_state_below,
                                                           recompute_references=recompute_references)
                            self.assertIsInstance(ds2, Dataset)

        # Final check: We update the val set with the test data. We check that dimensions match.
        split = 'val'
        len_test = 2996
        ds2 = update_dataset_from_file(copy.deepcopy(ds),
                                       params['DATA_ROOT_PATH'] + params['TEXT_FILES']['test'] + params['SRC_LAN'],
                                       params,
                                       splits=[split],
                                       output_text_filename=params['DATA_ROOT_PATH'] + params['TEXT_FILES']['test'] +
                                                            params['TRG_LAN'],
                                       remove_outputs=False,
                                       compute_state_below=True,
                                       recompute_references=True)
        self.assertIsInstance(ds2, Dataset)
        self.assertEqual(eval('ds2.len_' + split), len_test)
        self.assertTrue(eval('all(ds2.loaded_' + split + ')'))
        self.assertEqual(len(eval('ds2.X_' + split + str([params['INPUTS_IDS_DATASET'][0]]))), len_test)
        self.assertEqual(len(eval('ds2.Y_' + split + str([params['OUTPUTS_IDS_DATASET'][0]]))), len_test)


if __name__ == '__main__':
    unittest.main()
