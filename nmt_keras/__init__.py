__author__ = 'lvapeab'

import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(__name__)


def check_params(params):
    """
    Checks some typical parameters and warns if something wrong was specified.
    :param params: Model instance on which to apply the callback.
    :return: None
    """

    if params['SRC_PRETRAINED_VECTORS'] and params['SRC_PRETRAINED_VECTORS'][-4:] != '.npy':
        logger.warning('It seems that the pretrained word vectors provided for the target text are not in npy format.'
                       'You should preprocess the word embeddings with the "utils/preprocess_*_word_vectors.py script.')

    if params['TRG_PRETRAINED_VECTORS'] and params['TRG_PRETRAINED_VECTORS'][-4:] != '.npy':
        logger.warning('It seems that the pretrained word vectors provided for the target text are not in npy format.'
                       'You should preprocess the word embeddings with the "utils/preprocess_*_word_vectors.py script.')
    if not params['PAD_ON_BATCH']:
        logger.warning('It is HIGHLY recommended to set the option "PAD_ON_BATCH = True."')

    if params['MODEL_TYPE'].lower() == 'transformer':

        assert params['MODEL_SIZE'] == params['TARGET_TEXT_EMBEDDING_SIZE'], 'When using the Transformer model, ' \
                                                                             'dimensions of "MODEL_SIZE" and "TARGET_TEXT_EMBEDDING_SIZE" must match. ' \
                                                                             'Currently, they are: %d and %d, respectively.' % (
                                                                             params['MODEL_SIZE'],
                                                                             params['TARGET_TEXT_EMBEDDING_SIZE'])
        assert params['MODEL_SIZE'] == params['SOURCE_TEXT_EMBEDDING_SIZE'], 'When using the Transformer model, ' \
                                                                             'dimensions of "MODEL_SIZE" and "SOURCE_TEXT_EMBEDDING_SIZE" must match. ' \
                                                                             'Currently, they are: %d and %d, respectively.' % (
                                                                             params['MODEL_SIZE'],
                                                                             params['SOURCE_TEXT_EMBEDDING_SIZE'])

        if params['POS_UNK']:
            logger.warn('The "POS_UNK" option is still unimplemented for the "Transformer" model. Setting it to False.')
            params['POS_UNK'] = False
        assert params['MODEL_SIZE'] % params['N_HEADS'] == 0, \
            '"MODEL_SIZE" should be a multiple of "N_HEADS". ' \
            'Currently: mod(%d, %d) == %d.' % (
            params['MODEL_SIZE'], params['N_HEADS'], params['MODEL_SIZE'] % params['N_HEADS'])

    if params['POS_UNK']:
        if not params['OPTIMIZED_SEARCH']:
            logger.warn(
                'Unknown words replacement requires to use the optimized search ("OPTIMIZED_SEARCH" parameter). Setting "POS_UNK" to False.')
            params['POS_UNK'] = False

    if params['COVERAGE_PENALTY']:
        assert params['OPTIMIZED_SEARCH'], 'The application of "COVERAGE_PENALTY" requires ' \
                                           'to use the optimized search ("OPTIMIZED_SEARCH" parameter).'

    if 'from_logits' in params.get('LOSS', 'categorical_crossentropy'):
        if params.get('CLASSIFIER_ACTIVATION', 'softmax'):
            params['CLASSIFIER_ACTIVATION'] = None

    if params.get('LABEL_SMOOTHING', 0.) and 'sparse' in params.get('LOSS', 'categorical_crossentropy'):
        logger.warn('Label smoothing with sparse outputs is still unimplemented')

    if params.get('TRAIN_ONLY_LAST_LAYER'):
        logger.info('Training only last layer.')
        params['TRAINABLE_ENCODER'] = False
        params['TRAINABLE_DECODER'] = False

    return params
