from keras_wrapper.dataset import Dataset, saveDataset, loadDataset
import logging

logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(message)s', datefmt='%d/%m/%Y %H:%M:%S')


def update_dataset_from_file(ds,
                             input_text_filename,
                             params,
                             splits=list('val'),
                             output_text_filename=None,
                             remove_outputs=False,
                             compute_state_below=False):
    """
    Updates the dataset instance from a text file according to the given params.
    Used for sampling

    :param ds: Dataset instance
    :param input_text_filename: Source language sentences
    :param params: Parameters for building the dataset
    :param splits: Splits to sample
    :param output_text_filename: Target language sentences
    :param remove_outputs: Remove outputs from dataset (if True, will ignore the output_text_filename parameter)
    :param compute_state_below: Compute state below input (shifted target text for professor teaching)

    :return: Dataset object with the processed data
    """
    for split in splits:
        if remove_outputs:
            ds.removeOutput(split,
                            type='text',
                            id=params['OUTPUTS_IDS_DATASET'][0])
        elif output_text_filename is not None:
            ds.setOutput(output_text_filename,
                         split,
                         type='text',
                         id=params['OUTPUTS_IDS_DATASET'][0],
                         tokenization=params['TOKENIZATION_METHOD'],
                         build_vocabulary=False,
                         pad_on_batch=params['PAD_ON_BATCH'],
                         sample_weights=params['SAMPLE_WEIGHTS'],
                         max_text_len=params['MAX_OUTPUT_TEXT_LEN'],
                         max_words=params['OUTPUT_VOCABULARY_SIZE'],
                         min_occ=params['MIN_OCCURRENCES_OUTPUT_VOCAB'],
                         overwrite_split=True)

        # INPUT DATA
        ds.setInput(input_text_filename,
                    split,
                    type='text',
                    id=params['INPUTS_IDS_DATASET'][0],
                    pad_on_batch=params['PAD_ON_BATCH'],
                    tokenization=params['TOKENIZATION_METHOD'],
                    build_vocabulary=False,
                    fill=params['FILL'],
                    max_text_len=params['MAX_INPUT_TEXT_LEN'],
                    max_words=params['INPUT_VOCABULARY_SIZE'],
                    min_occ=params['MIN_OCCURRENCES_INPUT_VOCAB'],
                    overwrite_split=True)
        if compute_state_below:
            # INPUT DATA
            ds.setInput(output_text_filename,
                        split,
                        type='text',
                        id=params['INPUTS_IDS_DATASET'][1],
                        pad_on_batch=params['PAD_ON_BATCH'],
                        tokenization=params['TOKENIZATION_METHOD'],
                        build_vocabulary=False,
                        offset=1,
                        fill=params['FILL'],
                        max_text_len=params['MAX_INPUT_TEXT_LEN'],
                        max_words=params['INPUT_VOCABULARY_SIZE'],
                        min_occ=params['MIN_OCCURRENCES_OUTPUT_VOCAB'],
                        overwrite_split=True)
        else:
            ds.setInput(None,
                        split,
                        type='ghost',
                        id=params['INPUTS_IDS_DATASET'][-1],
                        required=False,
                        overwrite_split=True)

        if params['ALIGN_FROM_RAW']:
            ds.setRawInput(input_text_filename,
                           split,
                           type='file-name',
                           id='raw_' + params['INPUTS_IDS_DATASET'][0],
                           overwrite_split=True)

    return ds


def build_dataset(params):
    """
    Builds (or loads) a Dataset instance.
    :param params: Parameters specifying Dataset options
    :return: Dataset object
    """

    if params['REBUILD_DATASET']:  # We build a new dataset instance
        if params['VERBOSE'] > 0:
            silence = False
            logging.info(
                'Building ' + params['DATASET_NAME'] + '_' + params['SRC_LAN'] + params['TRG_LAN'] + ' dataset')
        else:
            silence = True

        base_path = params['DATA_ROOT_PATH']
        name = params['DATASET_NAME'] + '_' + params['SRC_LAN'] + params['TRG_LAN']
        ds = Dataset(name, base_path, silence=silence)

        # OUTPUT DATA
        # Let's load the train, val and test splits of the target language sentences (outputs)
        #    the files include a sentence per line.
        ds.setOutput(base_path + '/' + params['TEXT_FILES']['train'] + params['TRG_LAN'],
                     'train',
                     type='text',
                     id=params['OUTPUTS_IDS_DATASET'][0],
                     tokenization=params['TOKENIZATION_METHOD'],
                     build_vocabulary=True,
                     pad_on_batch=params['PAD_ON_BATCH'],
                     sample_weights=params['SAMPLE_WEIGHTS'],
                     max_text_len=params['MAX_OUTPUT_TEXT_LEN'],
                     max_words=params['OUTPUT_VOCABULARY_SIZE'],
                     min_occ=params['MIN_OCCURRENCES_OUTPUT_VOCAB'])
        if params['ALIGN_FROM_RAW'] and not params['HOMOGENEOUS_BATCHES']:
            ds.setRawOutput(base_path + '/' + params['TEXT_FILES']['train'] + params['TRG_LAN'],
                            'train',
                            type='file-name',
                            id='raw_' + params['OUTPUTS_IDS_DATASET'][0])

        for split in ['val', 'test']:
            if params['TEXT_FILES'].get(split) is not None:
                ds.setOutput(base_path + '/' + params['TEXT_FILES'][split] + params['TRG_LAN'],
                             split,
                             type='text',
                             id=params['OUTPUTS_IDS_DATASET'][0],
                             pad_on_batch=params['PAD_ON_BATCH'],
                             tokenization=params['TOKENIZATION_METHOD'],
                             sample_weights=params['SAMPLE_WEIGHTS'],
                             max_text_len=params['MAX_OUTPUT_TEXT_LEN'],
                             max_words=params['OUTPUT_VOCABULARY_SIZE'])
                if params['ALIGN_FROM_RAW'] and not params['HOMOGENEOUS_BATCHES']:
                    ds.setRawOutput(base_path + '/' + params['TEXT_FILES'][split] + params['TRG_LAN'],
                                    split,
                                    type='file-name',
                                    id='raw_' + params['OUTPUTS_IDS_DATASET'][0])

        # INPUT DATA
        # We must ensure that the 'train' split is the first (for building the vocabulary)
        for split in ['train', 'val', 'test']:
            if params['TEXT_FILES'].get(split) is not None:
                if split == 'train':
                    build_vocabulary = True
                else:
                    build_vocabulary = False
                ds.setInput(base_path + '/' + params['TEXT_FILES'][split] + params['SRC_LAN'],
                            split,
                            type='text',
                            id=params['INPUTS_IDS_DATASET'][0],
                            pad_on_batch=params['PAD_ON_BATCH'],
                            tokenization=params['TOKENIZATION_METHOD'],
                            build_vocabulary=build_vocabulary,
                            fill=params['FILL'],
                            max_text_len=params['MAX_INPUT_TEXT_LEN'],
                            max_words=params['INPUT_VOCABULARY_SIZE'],
                            min_occ=params['MIN_OCCURRENCES_INPUT_VOCAB'])

                if len(params['INPUTS_IDS_DATASET']) > 1:
                    if 'train' in split:
                        ds.setInput(base_path + '/' + params['TEXT_FILES'][split] + params['TRG_LAN'],
                                    split,
                                    type='text',
                                    id=params['INPUTS_IDS_DATASET'][1],
                                    required=False,
                                    tokenization=params['TOKENIZATION_METHOD'],
                                    pad_on_batch=params['PAD_ON_BATCH'],
                                    build_vocabulary=params['OUTPUTS_IDS_DATASET'][0],
                                    offset=1,
                                    fill=params['FILL'],
                                    max_text_len=params['MAX_OUTPUT_TEXT_LEN'],
                                    max_words=params['OUTPUT_VOCABULARY_SIZE'])
                    else:
                        ds.setInput(None,
                                    split,
                                    type='ghost',
                                    id=params['INPUTS_IDS_DATASET'][-1],
                                    required=False)
                if params['ALIGN_FROM_RAW'] and not params['HOMOGENEOUS_BATCHES']:
                    ds.setRawInput(base_path + '/' + params['TEXT_FILES'][split] + params['SRC_LAN'],
                                   split,
                                   type='file-name',
                                   id='raw_' + params['INPUTS_IDS_DATASET'][0])

        if params['POS_UNK']:
            if params['HEURISTIC'] > 0:
                ds.loadMapping(params['MAPPING'])

        # If we had multiple references per sentence
        keep_n_captions(ds, repeat=1, n=1, set_names=params['EVAL_ON_SETS'])

        # We have finished loading the dataset, now we can store it for using it in the future
        saveDataset(ds, params['DATASET_STORE_PATH'])

    else:
        # We can easily recover it with a single line
        ds = loadDataset(params['DATASET_STORE_PATH'] + '/Dataset_' + params['DATASET_NAME']
                         + '_' + params['SRC_LAN'] + params['TRG_LAN'] + '.pkl')

    return ds


def keep_n_captions(ds, repeat, n=1, set_names=None):
    """
    Keeps only n captions per image and stores the rest in dictionaries for a later evaluation
    :param ds: Dataset object
    :param repeat:
    :param n:
    :param set_names:
    :return:
    """

    n_samples = None
    X = None
    Y = None

    if set_names is None:
        set_names = ['val', 'test']
    for s in set_names:
        logging.info('Keeping ' + str(n) + ' captions per input on the ' + str(s) + ' set.')

        ds.extra_variables[s] = dict()
        exec ('n_samples = ds.len_' + s)

        # Process inputs
        for id_in in ds.ids_inputs:
            new_X = []
            if id_in in ds.optional_inputs:
                try:
                    exec ('X = ds.X_' + s)
                    for i in range(0, n_samples, repeat):
                        for j in range(n):
                            new_X.append(X[id_in][i + j])
                    exec ('ds.X_' + s + '[id_in] = new_X')
                except:
                    pass
            else:
                exec ('X = ds.X_' + s)
                for i in range(0, n_samples, repeat):
                    for j in range(n):
                        new_X.append(X[id_in][i + j])
                exec ('ds.X_' + s + '[id_in] = new_X')
        # Process outputs
        for id_out in ds.ids_outputs:
            new_Y = []
            exec ('Y = ds.Y_' + s)
            dict_Y = dict()
            count_samples = 0
            for i in range(0, n_samples, repeat):
                dict_Y[count_samples] = []
                for j in range(repeat):
                    if j < n:
                        new_Y.append(Y[id_out][i + j])
                    dict_Y[count_samples].append(Y[id_out][i + j])
                count_samples += 1
            exec ('ds.Y_' + s + '[id_out] = new_Y')
            # store dictionary with img_pos -> [cap1, cap2, cap3, ..., capN]
            ds.extra_variables[s][id_out] = dict_Y

        new_len = len(new_Y)
        exec ('ds.len_' + s + ' = new_len')
        logging.info('Samples reduced to ' + str(new_len) + ' in ' + s + ' set.')
