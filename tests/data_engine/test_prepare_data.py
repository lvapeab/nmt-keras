import pytest
import copy
from config import load_parameters
from data_engine.prepare_data import build_dataset, update_dataset_from_file, keep_n_captions
from keras_wrapper.dataset import Dataset, loadDataset


def test_build_datset():
    params = load_parameters()
    for verbose in range(2):
        params['REBUILD_DATASET'] = True
        params['VERBOSE'] = verbose
        params['DATASET_STORE_PATH'] = './'
        ds = build_dataset(params)
        assert isinstance(ds, Dataset)
        len_splits = [('train', 9900), ('val', 100), ('test', 2996)]
        for split, len_split in len_splits:
            assert eval('ds.len_' + split) == len_split
            assert eval('all(ds.loaded_' + split + ')')
            assert len(eval('ds.X_' + split + str([params['INPUTS_IDS_DATASET'][0]]))) == len_split
            assert len(eval('ds.Y_' + split + str([params['OUTPUTS_IDS_DATASET'][0]]))) == len_split


def test_load_dataset():
    params = load_parameters()
    ds = loadDataset('./Dataset_' + params['DATASET_NAME'] + '_' + params['SRC_LAN'] + params['TRG_LAN'] + '.pkl')
    assert isinstance(ds, Dataset)
    assert isinstance(ds.vocabulary, dict)
    assert len(list(ds.vocabulary)) >= 3
    for voc in ds.vocabulary:
        assert len(list(ds.vocabulary[voc])) == 2


def test_update_dataset_from_file():
    params = load_parameters()
    for rebuild_dataset in [True, False]:
        params['REBUILD_DATASET'] = rebuild_dataset
        params['DATASET_STORE_PATH'] = './'
        for splits in [[], None, ['val']]:
            ds = build_dataset(params)
            assert isinstance(ds, Dataset)
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
                            assert isinstance(ds2, Dataset)

    # Final check: We update the val set with the test data. We check that dimensions match.
    split = 'val'
    len_test = 2996
    ds2 = update_dataset_from_file(copy.deepcopy(ds),
                                   params['DATA_ROOT_PATH'] + params['TEXT_FILES']['test'] + params['SRC_LAN'],
                                   params,
                                   splits=[split],
                                   output_text_filename=params['DATA_ROOT_PATH'] + params['TEXT_FILES']['test'] + params['TRG_LAN'],
                                   remove_outputs=False,
                                   compute_state_below=True,
                                   recompute_references=True)
    assert isinstance(ds2, Dataset)
    assert eval('ds2.len_' + split) == len_test
    assert eval('all(ds2.loaded_' + split + ')')
    assert len(eval('ds2.X_' + split + str([params['INPUTS_IDS_DATASET'][0]]))) == len_test
    assert len(eval('ds2.Y_' + split + str([params['OUTPUTS_IDS_DATASET'][0]]))) == len_test

    if __name__ == '__main__':
        pytest.main([__file__])


def test_keep_n_captions():
    params = load_parameters()
    params['REBUILD_DATASET'] = True
    params['DATASET_STORE_PATH'] = './'
    ds = build_dataset(params)
    len_splits = {'train': 9900, 'val': 100, 'test': 2996}

    for splits in [[], None, ['val'], ['val', 'test']]:
        keep_n_captions(ds, 1, n=1, set_names=splits)
        if splits is not None:
            for split in splits:
                len_split = len_splits[split]
                assert eval('ds.len_' + split) == len_split
                assert eval('all(ds.loaded_' + split + ')')
                assert len(eval('ds.X_' + split + str([params['INPUTS_IDS_DATASET'][0]]))) == len_split
                assert len(eval('ds.Y_' + split + str([params['OUTPUTS_IDS_DATASET'][0]]))) == len_split

    if __name__ == '__main__':
        pytest.main([__file__])
