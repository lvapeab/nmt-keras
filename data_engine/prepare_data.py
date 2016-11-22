from keras_wrapper.dataset import Dataset, saveDataset, loadDataset

import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(message)s', datefmt='%d/%m/%Y %H:%M:%S')

def build_dataset(params):
    
    if params['REBUILD_DATASET']: # We build a new dataset instance
        if(params['VERBOSE'] > 0):
            silence=False
            logging.info('Building ' + params['DATASET_NAME'] + ' dataset')
        else:
            silence=True

        base_path = params['DATA_ROOT_PATH']
        name = params['DATASET_NAME']
        ds = Dataset(name, base_path, silence=silence)

        ##### OUTPUT DATA
        # Let's load the train, val and test splits of the target language sentences (outputs)
        #    the files include a sentence per line.

        ds.setOutput(base_path+'/'+params['TEXT_FILES']['train']+params['TRG_LAN'], 'train',
                   type='text', id=params['OUTPUTS_IDS_DATASET'][0],
                   tokenization=params['TOKENIZATION_METHOD'], build_vocabulary=True, pad_on_batch=True,
                   sample_weights=params['SAMPLE_WEIGHTS'],
                   max_text_len=params['MAX_OUTPUT_TEXT_LEN'], max_words=params['OUTPUT_VOCABULARY_SIZE'],
                   min_occ=params['MIN_OCCURRENCES_VOCAB'])

        ds.setOutput(base_path+'/'+params['TEXT_FILES']['val']+params['TRG_LAN'], 'val',
                   type='text', id=params['OUTPUTS_IDS_DATASET'][0], pad_on_batch=True,
                   tokenization=params['TOKENIZATION_METHOD'],
                   sample_weights=params['SAMPLE_WEIGHTS'],
                   max_text_len=params['MAX_OUTPUT_TEXT_LEN'], max_words=params['OUTPUT_VOCABULARY_SIZE'])

        ds.setOutput(base_path+'/'+params['TEXT_FILES']['test']+params['TRG_LAN'], 'test',
                   type='text', id=params['OUTPUTS_IDS_DATASET'][0], pad_on_batch=True,
                   tokenization=params['TOKENIZATION_METHOD'],
                   sample_weights=params['SAMPLE_WEIGHTS'],
                   max_text_len=params['MAX_OUTPUT_TEXT_LEN'], max_words=params['OUTPUT_VOCABULARY_SIZE'])

        # INPUT DATA
        for split in ['train', 'val', 'test']:
            if params['TEXT_FILES'].get(split) is not None:
                if split == 'train':
                    build_vocabulary = True
                else:
                    build_vocabulary = False

                ds.setInput(base_path+'/'+params['TEXT_FILES'][split]+params['SRC_LAN'], split,
                           type='text', id=params['INPUTS_IDS_DATASET'][0], pad_on_batch=True,
                           tokenization=params['TOKENIZATION_METHOD'], build_vocabulary=build_vocabulary,
                           fill=params['FILL'], max_text_len=params['MAX_INPUT_TEXT_LEN'],
                           max_words=params['INPUT_VOCABULARY_SIZE'], min_occ=params['MIN_OCCURRENCES_VOCAB'])

                if len(params['INPUTS_IDS_DATASET']) > 1:
                    if split == 'train':
                        ds.setInput(base_path+'/'+params['TEXT_FILES'][split]+params['TRG_LAN'], split,
                                   type='text', id=params['INPUTS_IDS_DATASET'][1], required=False,
                                   tokenization=params['TOKENIZATION_METHOD'], pad_on_batch=True,
                                   build_vocabulary=params['OUTPUTS_IDS_DATASET'][0], offset=1,
                                   fill=params['FILL'],
                                   max_text_len=params['MAX_OUTPUT_TEXT_LEN'], max_words=params['OUTPUT_VOCABULARY_SIZE'])
                    else:
                        ds.setInput(None, split, type='ghost', id=params['INPUTS_IDS_DATASET'][-1], required=False)


        keep_n_captions(ds, repeat=1, n=1, set_names=['val','test'])

        # We have finished loading the dataset, now we can store it for using it in the future
        saveDataset(ds, params['DATASET_STORE_PATH'])

    
    else:
        # We can easily recover it with a single line
        ds = loadDataset(params['DATASET_STORE_PATH']+'/Dataset_'+params['DATASET_NAME']+'.pkl')

    return ds

def keep_n_captions(ds, repeat, n=1, set_names=['val','test']):
    ''' Keeps only n captions per image and stores the rest in dictionaries for a later evaluation
    '''

    for s in set_names:
        logging.info('Keeping '+str(n)+' captions per input on the '+str(s)+' set.')

        ds.extra_variables[s] = dict()
        exec('n_samples = ds.len_'+s)

        # Process inputs
        for id_in in ds.ids_inputs:
            new_X = []
            if id_in in ds.optional_inputs:
                try:
                    exec('X = ds.X_'+s)
                    for i in range(0, n_samples, repeat):
                        for j in range(n):
                            new_X.append(X[id_in][i+j])
                    exec('ds.X_'+s+'[id_in] = new_X')
                except: pass
            else:
                exec('X = ds.X_'+s)
                for i in range(0, n_samples, repeat):
                    for j in range(n):
                        new_X.append(X[id_in][i+j])
                exec('ds.X_'+s+'[id_in] = new_X')
        # Process outputs
        for id_out in ds.ids_outputs:
            new_Y = []
            exec('Y = ds.Y_'+s)
            dict_Y = dict()
            count_samples = 0
            for i in range(0, n_samples, repeat):
                dict_Y[count_samples] = []
                for j in range(repeat):
                    if(j < n):
                        new_Y.append(Y[id_out][i+j])
                    dict_Y[count_samples].append(Y[id_out][i+j])
                count_samples += 1
            exec('ds.Y_'+s+'[id_out] = new_Y')
            # store dictionary with img_pos -> [cap1, cap2, cap3, ..., capN]
            ds.extra_variables[s][id_out] = dict_Y

        new_len = len(new_Y)
        exec('ds.len_'+s+' = new_len')
        logging.info('Samples reduced to '+str(new_len)+' in '+s+' set.')

if __name__ == "__main__":
    params = dict()
    
    # Parameters (this should be externally provided from a config file)
    params['RELOAD_DATASET'] = True # build again or use stored instance
    params['DATASET_NAME'] = 'Translation_toy'
    params['DATA_ROOT_PATH'] = '/media/HDD_2TB/DATASETS/Translation_toy'
    params['TOKENIZATION_METHOD'] = 'tokenize_basic'

    params['TEXT_FILES'] = {'train': 'training.', 'val': 'val.', 'test': 'test.'}
    params['INPUTS_IDS_DATASET'] = ['source_text']
    params['OUTPUTS_IDS_DATASET'] = ['target_text']
    
    params['MAX_INPUT_TEXT_LEN'] = 35
    params['INPUT_VOCABULARY_SIZE'] = 0
    
    params['MAX_OUTPUT_TEXT_LEN'] = 35
    params['OUTPUT_VOCABULARY_SIZE'] = 0
    
    params['SRC_LAN'] = 'en'
    params['TRG_LAN'] = 'es'
    
    
    params['VERBOSE'] = 1
        
    ds = build_dataset(params)


    """
    # Lets recover the first batch of data
    [X, Y] = ds.getXY('train', 10)

    logging.info('Processed data:')
    print X
    print Y
    print
    logging.info('Unprocessed data:')
    ds.resetCounters('train')
    [X, Y] = ds.getXY('train', 10, debug=True)
    print X
    print Y
    print
    """

    logging.info('Sample data loaded correctly.')
    print
