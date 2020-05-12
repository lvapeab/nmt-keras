import logging
import os
from keras_wrapper.dataset import Dataset, saveDataset, loadDataset

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(__name__)


def update_dataset_from_file(ds,
                             input_text_filename,
                             params,
                             splits=None,
                             output_text_filename=None,
                             remove_outputs=False,
                             compute_state_below=False,
                             recompute_references=False):
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
    :param recompute_references: Whether we should rebuild the references of the dataset or not.

    :return: Dataset object with the processed data
    """
    if splits is None:
        splits = ['val']

    if output_text_filename is None:
        recompute_references = False

    for split in splits:
        if split == 'train':
            output_type = params.get('OUTPUTS_TYPES_DATASET', ['dense-text'] if 'sparse' in params['LOSS'] else ['text'])[0]
        else:
            # Type of val/test outuput is always 'text' or 'dense-text'
            output_type = 'dense-text' if 'sparse' in params['LOSS'] else 'text'

        if remove_outputs:
            ds.removeOutput(split,
                            id=params['OUTPUTS_IDS_DATASET'][0])
            recompute_references = False

        elif output_text_filename is not None:
            ds.setOutput(output_text_filename,
                         split,
                         type=output_type,
                         id=params['OUTPUTS_IDS_DATASET'][0],
                         tokenization=params.get('TOKENIZATION_METHOD', 'tokenize_none'),
                         build_vocabulary=False,
                         pad_on_batch=params.get('PAD_ON_BATCH', True),
                         fill=params.get('FILL', 'end'),
                         sample_weights=params.get('SAMPLE_WEIGHTS', True),
                         max_text_len=params.get('MAX_OUTPUT_TEXT_LEN', 100),
                         max_words=params.get('OUTPUT_VOCABULARY_SIZE', 0),
                         min_occ=params.get('MIN_OCCURRENCES_OUTPUT_VOCAB', 0),
                         bpe_codes=params.get('BPE_CODES_PATH', None),
                         label_smoothing=params.get('LABEL_SMOOTHING', 0.),
                         overwrite_split=True)

        # INPUT DATA
        ds.setInput(input_text_filename,
                    split,
                    type=params.get('INPUTS_TYPES_DATASET', ['text', 'text'])[0],
                    id=params['INPUTS_IDS_DATASET'][0],
                    tokenization=params.get('TOKENIZATION_METHOD', 'tokenize_none'),
                    build_vocabulary=False,
                    pad_on_batch=params.get('PAD_ON_BATCH', True),
                    fill=params.get('FILL', 'end'),
                    max_text_len=params.get('MAX_INPUT_TEXT_LEN', 100),
                    max_words=params.get('INPUT_VOCABULARY_SIZE', 0),
                    min_occ=params.get('MIN_OCCURRENCES_INPUT_VOCAB', 0),
                    bpe_codes=params.get('BPE_CODES_PATH', None),
                    overwrite_split=True)

        if compute_state_below and output_text_filename is not None:
            # INPUT DATA
            ds.setInput(output_text_filename,
                        split,
                        type=params.get('INPUTS_TYPES_DATASET', ['text', 'text'])[1],
                        id=params['INPUTS_IDS_DATASET'][1],
                        pad_on_batch=params.get('PAD_ON_BATCH', True),
                        tokenization=params.get('TOKENIZATION_METHOD', 'tokenize_none'),
                        build_vocabulary=False,
                        offset=1,
                        fill=params.get('FILL', 'end'),
                        max_text_len=params.get('MAX_OUTPUT_TEXT_LEN', 100),
                        max_words=params.get('OUTPUT_VOCABULARY_SIZE', 0),
                        min_occ=params.get('MIN_OCCURRENCES_OUTPUT_VOCAB', 0),
                        bpe_codes=params.get('BPE_CODES_PATH', None),
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

        # If we had multiple references per sentence
        if recompute_references:
            prepare_references(ds, repeat=1, n=1, set_names=params['EVAL_ON_SETS'])

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
            logger.info('Building ' + params['DATASET_NAME'] + '_' + params['SRC_LAN'] + params['TRG_LAN'] + ' dataset')
        else:
            silence = True

        base_path = params['DATA_ROOT_PATH']
        name = params['DATASET_NAME'] + '_' + params['SRC_LAN'] + params['TRG_LAN']
        ds = Dataset(name, base_path, silence=silence)

        # OUTPUT DATA
        # Load the train, val and test splits of the target language sentences (outputs). The files include a sentence per line.
        ds.setOutput(os.path.join(base_path, params['TEXT_FILES']['train'] + params['TRG_LAN']),
                     'train',
                     type=params.get('OUTPUTS_TYPES_DATASET',
                                     ['dense-text'] if 'sparse' in params['LOSS'] else ['text'])[0],
                     id=params['OUTPUTS_IDS_DATASET'][0],
                     tokenization=params.get('TOKENIZATION_METHOD', 'tokenize_none'),
                     build_vocabulary=True,
                     pad_on_batch=params.get('PAD_ON_BATCH', True),
                     sample_weights=params.get('SAMPLE_WEIGHTS', True),
                     fill=params.get('FILL', 'end'),
                     max_text_len=params.get('MAX_OUTPUT_TEXT_LEN', 70),
                     max_words=params.get('OUTPUT_VOCABULARY_SIZE', 0),
                     min_occ=params.get('MIN_OCCURRENCES_OUTPUT_VOCAB', 0),
                     bpe_codes=params.get('BPE_CODES_PATH', None),
                     label_smoothing=params.get('LABEL_SMOOTHING', 0.))

        for split in ['val', 'test']:
            if params['TEXT_FILES'].get(split) is not None:
                ds.setOutput(os.path.join(base_path, params['TEXT_FILES'][split] + params['TRG_LAN']),
                             split,
                             type='text',  # The type of the references should be always 'text'
                             id=params['OUTPUTS_IDS_DATASET'][0],
                             pad_on_batch=params.get('PAD_ON_BATCH', True),
                             tokenization=params.get('TOKENIZATION_METHOD', 'tokenize_none'),
                             sample_weights=params.get('SAMPLE_WEIGHTS', True),
                             max_text_len=params.get('MAX_OUTPUT_TEXT_LEN', 70),
                             max_words=params.get('OUTPUT_VOCABULARY_SIZE', 0),
                             bpe_codes=params.get('BPE_CODES_PATH', None),
                             label_smoothing=0.)

        # INPUT DATA
        # We must ensure that the 'train' split is the first (for building the vocabulary)
        for split in params['TEXT_FILES']:
            build_vocabulary = split == 'train'
            ds.setInput(os.path.join(base_path, params['TEXT_FILES'][split] + params['SRC_LAN']),
                        split,
                        type=params.get('INPUTS_TYPES_DATASET', ['text', 'text'])[0],
                        id=params['INPUTS_IDS_DATASET'][0],
                        pad_on_batch=params.get('PAD_ON_BATCH', True),
                        tokenization=params.get('TOKENIZATION_METHOD', 'tokenize_none'),
                        build_vocabulary=build_vocabulary,
                        fill=params.get('FILL', 'end'),
                        max_text_len=params.get('MAX_INPUT_TEXT_LEN', 70),
                        max_words=params.get('INPUT_VOCABULARY_SIZE', 0),
                        min_occ=params.get('MIN_OCCURRENCES_INPUT_VOCAB', 0),
                        bpe_codes=params.get('BPE_CODES_PATH', None))

            if len(params['INPUTS_IDS_DATASET']) > 1:
                if 'train' in split:
                    ds.setInput(os.path.join(base_path, params['TEXT_FILES'][split] + params['TRG_LAN']),
                                split,
                                type=params.get('INPUTS_TYPES_DATASET', ['text', 'text'])[1],
                                id=params['INPUTS_IDS_DATASET'][1],
                                required=False,
                                tokenization=params.get('TOKENIZATION_METHOD', 'tokenize_none'),
                                pad_on_batch=params.get('PAD_ON_BATCH', True),
                                build_vocabulary=params['OUTPUTS_IDS_DATASET'][0],
                                offset=1,
                                fill=params.get('FILL', 'end'),
                                max_text_len=params.get('MAX_OUTPUT_TEXT_LEN', 70),
                                max_words=params.get('OUTPUT_VOCABULARY_SIZE', 0),
                                bpe_codes=params.get('BPE_CODES_PATH', None))
                    if params.get('TIE_EMBEDDINGS', False):
                        ds.merge_vocabularies([params['INPUTS_IDS_DATASET'][1], params['INPUTS_IDS_DATASET'][0]])
                else:
                    ds.setInput(None,
                                split,
                                type='ghost',
                                id=params['INPUTS_IDS_DATASET'][-1],
                                required=False)
            if params.get('ALIGN_FROM_RAW', True) and not params.get('HOMOGENEOUS_BATCHES', False):
                ds.setRawInput(os.path.join(base_path, params['TEXT_FILES'][split] + params['SRC_LAN']),
                               split,
                               type='file-name',
                               id='raw_' + params['INPUTS_IDS_DATASET'][0])
        if params.get('POS_UNK', False):
            if params.get('HEURISTIC', 0) > 0:
                ds.loadMapping(params['MAPPING'])
        # Prepare references
        prepare_references(ds,
                           repeat=1,
                           n=1,
                           set_names=params['EVAL_ON_SETS'])

        # We have finished loading the dataset, now we can store it for using it in the future
        saveDataset(ds, params['DATASET_STORE_PATH'])

    else:
        # We can easily recover it with a single line
        ds = loadDataset(os.path.join(params['DATASET_STORE_PATH'],
                                      'Dataset_' + params['DATASET_NAME'] +
                                      '_' + params['SRC_LAN'] + params['TRG_LAN'] + '.pkl'))

        # Prepare references
        prepare_references(ds,
                           repeat=1,
                           n=1,
                           set_names=params['EVAL_ON_SETS'])

    return ds


def prepare_references(ds, repeat, n=1, set_names=None):
    """
    Keeps only n captions per image and stores the rest in dictionaries for a later evaluation
    :param ds: Dataset object
    :param repeat: Number of input samples per output
    :param n: Number of outputs to keep.
    :param set_names: Set name.
    :return:
    """

    if set_names is None:
        set_names = ['val', 'test']
    for s in set_names:
        logger.info('Keeping ' + str(n) + ' captions per input on the ' + str(s) + ' set.')

        ds.extra_variables[s] = dict()
        n_samples = getattr(ds, 'len_' + s)
        # Process inputs
        for id_in in ds.ids_inputs:
            new_X = []
            if id_in in ds.optional_inputs:
                try:
                    X = getattr(ds, 'X_' + s)
                    for i in range(0, n_samples, repeat):
                        for j in range(n):
                            new_X.append(X[id_in][i + j])
                    setattr(ds, 'X_' + s + '[' + id_in + ']', new_X)
                except Exception:
                    pass
            else:
                X = getattr(ds, 'X_' + s)
                for i in range(0, n_samples, repeat):
                    for j in range(n):
                        new_X.append(X[id_in][i + j])
                aux_list = getattr(ds, 'X_' + s)
                aux_list[id_in] = new_X
                setattr(ds, 'X_' + s, aux_list)
                del aux_list
        # Process outputs
        for id_out in ds.ids_outputs:
            new_Y = []
            Y = getattr(ds, 'Y_' + s)
            dict_Y = dict()
            count_samples = 0
            for i in range(0, n_samples, repeat):
                dict_Y[count_samples] = []
                for j in range(repeat):
                    if j < n:
                        new_Y.append(Y[id_out][i + j])
                    dict_Y[count_samples].append(Y[id_out][i + j])
                count_samples += 1

            aux_list = getattr(ds, 'Y_' + s)
            aux_list[id_out] = new_Y
            setattr(ds, 'Y_' + s, aux_list)
            del aux_list

            # store dictionary with img_pos -> [cap1, cap2, cap3, ..., capN]
            ds.extra_variables[s][id_out] = dict_Y

        new_len = len(new_Y)
        setattr(ds, 'len_' + s, new_len)

        logger.info('Samples reduced to ' + str(new_len) + ' in ' + s + ' set.')

# Backwards compatibility:
keep_n_captions = prepare_references
