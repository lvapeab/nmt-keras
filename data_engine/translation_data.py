import logging
import os

import numpy as np
from toolz import compose
from toolz import frequencies

from keras.preprocessing import sequence
from utils.common import preprocess_line, PADDING, EXTRA_WORDS, UNKNOWN, EOA, EOQ, index_sequence, build_vocabulary

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def Caption_Data(root_path='./', split='train', dataset_type='flickr30k', src_lan='en', trg_lan='de'):

    """
    Multimodal translation data provider
    :param root_path:
    :param split:
    :param dataset_type:
    :return:
    """

    src_corpus = map(lambda x: x[: -1], open(os.path.join(root_path, '{0}.{1}'.format(split, src_lan))).readlines())
    trg_corpus = map(lambda x: x[: -1], open(os.path.join(root_path, '{0}.{1}'.format(split, trg_lan))).readlines())

    x_list = []
    y_list = []
    # return only source sentences if there are no targets
    if split == 'test':
        for i in range(len(src_corpus)):
            x_list.append(src_corpus[i])
        return {'x': x_list, 'y': None}

    for i in range(len(src_corpus)):
        caption_src = preprocess_line(src_corpus[i])
        caption_trg = preprocess_line(trg_corpus[i])
        x_list.append(caption_src)
        y_list.append(caption_trg)
    return {'x': x_list, 'y': y_list}


def encode_captions_index(x, word2index_x, max_time_steps=None):
    """
    Index-based encoding of questions.

    In:
        x - list of questions
        word2index_x - mapping from question words to indices (inverted vocabulary)
        max_time_steps - maximal number of words in the question (max. time steps);
            if None then all question words are taken;
            by default None
    Out:
        a list of encoded questions
    """
    x_modified = [q + ' ' + EOQ for q in x]
    if max_time_steps is not None:
        x_modified = [' '.join(q.split()[:max_time_steps]) for q in x]
    return index_sequence(x_modified, word2index_x)


def encode_captions_src_one_hot(x, word2index_x, max_time_steps):
    """
    One-hot encoding of questions.

    In:
        x - list of  questions
        word2index_x - mapping from question words to indices (inverted vocabulary)
        max_time_steps - maximal number of words in the sequence (max. time steps)

    Out:
        boolean tensor of size: data_size x max_time_steps x vocabulary_size
            for a given question and a time step there is only one '1'
    """
    X = np.zeros((len(x), max_time_steps, len(word2index_x.keys())),
            dtype=np.bool)
    # encode questions
    for question_no, question in enumerate(x):
        question_word_list = question.split()
        question_word_list.append(EOQ)
        for word_no, word in enumerate(question_word_list):
            word = word.strip()
            if word_no == max_time_steps - 1:
                # we need to finish
                this_index = word2index_x[EOQ]
            else:
                if word in word2index_x:
                    this_index = word2index_x[word]
                else:
                    this_index = word2index_x[UNKNOWN]
            X[question_no, word_no, this_index] = 1
    return X

def encode_captions_dense(x, word_encoder, max_time_steps):
    """
    Dense representation of questions.

    In:
        x - list of questions
        word_encoder - encodes words
        max_time_steps - maximal number of words in the sequence (max. time steps)
        is_remove_question_symbol - true if we remove question symbols from the questions;
            by default it is False

    Out:
        float tensor of size: data_size x max_time_steps x dense_encoding_size
    """
    word_encoder_dim = word_encoder(unicode(x[0].split()[0].strip())).vector.shape[0]
    X = np.zeros((len(x), max_time_steps, word_encoder_dim))
    for question_no, question in enumerate(x):
        question_word_list = question.split()
        reversed_question_word_list = question_word_list[::-1]
        for word_no, raw_word in enumerate(reversed_question_word_list):
            word = unicode(raw_word.strip())
            this_representation = word_encoder(word).vector
            if max_time_steps - word_no - 1 >= 0:
                X[question_no, max_time_steps - word_no - 1, :] = this_representation
            else:
                break
    return X


def encode_trg_one_hot(y, word2index_y, max_time_steps=10):
    """
    One-hot encoding of answers.
    If more than first answer word is encoded then the answer words
    are modelled as sequence.

    In:
        y - list of answers
        word2index_y - mapping from answer words to indices (vocabulary)
        max_time_steps - maximal number of words in the sequence (max. time steps)
            by default 10
        is_only_first_answer_word - if True then only first answer word is taken
            by default False
        answer_words_delimiter - a symbol for splitting answer into answer words;
            if None is provided then we don't split answer into answer words
            (that is the whole answer is an answer word);
            by default ','

    Out:
        Y - boolean matrix of size:
                data_size x vocabulary_size if there is only single answer word
                data_size x max_time_steps x vocabulary_size otherwise
                    the matrix is padded
            for a given answer and a time step there is only one '1'
        y_gt - list of answers
            the same as input 'y' if is_only_first_answer_word==False
            only first words from 'y' if is_only_first_answer_word==True
    """
    # encode answers
    Y = np.zeros((len(y), max_time_steps, len(word2index_y.keys())), dtype=np.bool)
    y_index = np.zeros((len(y), max_time_steps), dtype=np.int32)
    y_gt = y
    for answer_no, answer in enumerate(y):
        answer_split = answer.split(' ')
        for word_no, word in enumerate(answer_split):
            word = word.strip()
            if word_no == max_time_steps - 1:
                break
            if word in word2index_y:
                Y[answer_no, word_no, word2index_y[word]] = 1
                y_index[answer_no, word_no] = word2index_y[word]
            else:
                Y[answer_no, word_no, word2index_y[UNKNOWN]] = 1
                y_index[answer_no, word_no] = word2index_y[UNKNOWN]
        Y[answer_no, min(len(answer_split), max_time_steps - 1), word2index_y[EOA]] = 1
        y_index[answer_no, min(len(answer_split), max_time_steps - 1)] = word2index_y[EOA]

    return Y, y_index, y_gt




def encode_trg_indices(y, word2index_y, max_time_steps=10):
    """
    One-hot encoding of answers.
    If more than first answer word is encoded then the answer words
    are modelled as sequence.

    In:
        y - list of answers
        word2index_y - mapping from answer words to indices (vocabulary)
        max_time_steps - maximal number of words in the sequence (max. time steps)
            by default 10
        is_only_first_answer_word - if True then only first answer word is taken
            by default False
        answer_words_delimiter - a symbol for splitting answer into answer words;
            if None is provided then we don't split answer into answer words
            (that is the whole answer is an answer word);
            by default ','

    Out:
        Y - boolean matrix of size:
                data_size x vocabulary_size if there is only single answer word
                data_size x max_time_steps x vocabulary_size otherwise
                    the matrix is padded
            for a given answer and a time step there is only one '1'
        y_gt - list of answers
            the same as input 'y' if is_only_first_answer_word==False
            only first words from 'y' if is_only_first_answer_word==True
    """
    # encode answers
    Y = np.zeros((len(y), max_time_steps), dtype='int64')
    y_gt = y

    for answer_no, answer in enumerate(y):
        answer_split = answer.split(' ')
        for word_no, word in enumerate(answer_split):
            word = word.strip()
            if word_no == max_time_steps - 1:
                break
            if word in word2index_y:
                Y[answer_no, word_no] =  word2index_y[word]
            else:
                Y[answer_no, word_no] =  word2index_y[UNKNOWN]
        Y[answer_no, min(len(answer_split), max_time_steps - 1)] = word2index_y[EOA]
    return Y, y_gt




def get_Multimoda_Translation_datasets(params, splits=['train', 'val', 'test'], vocabularies=None):
    """
        Loads all the splits specified by the parameter 'splits'. If 'vocabularies' is not specified then the split
        'train' should be included in 'splits' and additionally it should be in the first position in order to start
        computing the necessary vocabularies.
        
        Returns each of the splits, each of them in a dict with the following information:
            ['X'] X : [ numpy.array(img_feat)  ,  numpy.array(src_ids)  ]
            ['Y'] Y : numpy.array(ans_classes)
            ['X_text'] X_text  (visualization purposes)  :  src_text
            ['Y_text'] Y_text (visualization purposes) :  ans_text
            # extra information for evaluation
            ['vqa_object'] vqa_object : used for evaluation
            ['questions_path'] questions_path : used for evaluation
            ['questions_id_vqa'] questions_id_vqa: used for evaluation
        Additionally, returns the produced vocabularies.
            
    """
    
    data = []
    
    if(not 'train' in splits):
        logger.warning('\tNot loading training set')

    for split in splits:
        
        # Load data
        if params['TRAIN_ON_TRAINVAL']:
            logging.info("Training jointly on train and val sets.")
            x_src_text_t, y_trg_text_t = load_data(params, split='train')
            x_src_text_v, y_trg_text_v = load_data(params, split='val')
            x_src_text = x_src_text_t + x_src_text_v
            y_trg_text = y_trg_text_t + y_trg_text_v
        else:
            x_src_text, y_trg_text = load_data(params, split=split)
        
        # Build vocabularies
        if((split == 'train' and vocabularies is None) or (split == 'train' and params['FORCE_RELOAD_VOCABULARY'])):
            word2index_src, word2index_trg, index2word_src, index2word_trg = build_vocabularies(x_src_text, y_trg_text, params)
            # We additionally store the produced vocabularies
            vocabularies = dict()
            vocabularies['word2index_src'] = word2index_src
            vocabularies['word2index_trg'] = word2index_trg
            vocabularies['index2word_src'] = index2word_src
            vocabularies['index2word_trg'] = index2word_trg
        else:
            word2index_src = vocabularies['word2index_src']
            word2index_trg = vocabularies['word2index_trg']
        
        # Encode data
        src_ids, trg_ids, next_words = encode_data(x_src_text, y_trg_text, word2index_src, word2index_trg, params)
        
        # Stores the following information in each position of the dict
        d = dict()
        # ['X'] X : [ numpy.array(img_feat)  ,  numpy.array(qst_ids)  ]
        d['X'] = np.array(src_ids)
        d['state_below'] = np.array(next_words)
        # ['Y'] Y : numpy.array(trg_classes)
        d['Y'] = trg_ids # np.array(trg_classes)
        # ['X_text'] X_text  (visualization purposes)  :  qst_text 
        d['X_text'] = x_src_text
        # ['Y_text'] Y_text (visualization purposes) :  trg_text
        d['Y_text'] = y_trg_text
        # extra information
        data.append(d)

    return data, vocabularies



def load_data(params, split='train'):
    
    if(split=='train' or params['TRAIN_ON_TRAINVAL']):
        shuffle = True
    else:
        shuffle = False

    logger.info('\tLoading '+split+' captions')
    dataset = Caption_Data(root_path=params['DATA_ROOT_PATH'], split=split, dataset_type='flickr30k',
                           src_lan=params['SRC_LAN'], trg_lan=params['TRG_LAN'])
                                
    captions_src, captions_trg = dataset['x'], dataset['y']
    logger.debug('\t\tNumber of '+ split +' examples {0}'.format(len(captions_src)))
    return captions_src, captions_trg



def build_vocabularies(X, Y, params):
    
    logger.info('\tBuilding vocabularies')
    
    split_symbol = '{'
    if type(X[0]) is unicode:
        # choose a split symbol that doesn't exist in text
        split_function = lambda x: unicode.split(x, split_symbol)
    elif type(X[0]) is str:
        split_function = lambda x: str.split(x, split_symbol)
    else:
        raise NotImplementedError()

    wordcount = compose(frequencies, split_function)
    wordcount_x = wordcount(split_symbol.join(X).replace(' ',split_symbol))
    wordcount_y = wordcount(split_symbol.join(Y).replace(' ',split_symbol))

    word2index_x, index2word_x = build_vocabulary(
            this_wordcount=wordcount_x,
            is_reset=True,
            truncate_to_most_frequent=params['INPUT_VOCABULARY_SIZE'])
    word2index_y, index2word_y = build_vocabulary(
            this_wordcount=wordcount_y,
            is_reset=True,
            truncate_to_most_frequent=params['OUTPUT_VOCABULARY_SIZE'])

    index2word_x = {v: k for k, v in word2index_x.items()}
    index2word_y = {v: k for k, v in word2index_y.items()}

    logger.info('\tSize of the input {0}, and output vocabularies {1}'.format(len(word2index_x), len(word2index_y)))

    return word2index_x, word2index_y, index2word_x, index2word_y


def encode_data(captions_src, captions_trg, word2index_src, word2index_trg, params):
    
    # SRC LANG
    if params['WORD_REPRESENTATION'] == 'one_hot':
        one_hot_x = encode_captions_index(captions_src, word2index_src)
        src_ids = sequence.pad_sequences(one_hot_x, maxlen=params['MAX_TIME_STEPS'])
    elif params['WORD_REPRESENTATION'] == 'dense':
        raise NotImplementedError()
    else:
        raise NotImplementedError()
    # TRG LANG
    #one_hot_y = encode_captions_index(captions_trg, word2index_trg)
    #trg_ids = sequence.pad_sequences(one_hot_y, maxlen=params['MAX_TIME_STEPS'])
    trg_ids, trg_indices, trg_classes = encode_trg_one_hot(captions_trg, word2index_trg, max_time_steps=params['MAX_TIME_STEPS'])
    next_words = np.append([EXTRA_WORDS[PADDING]]*trg_indices.shape[0], trg_indices[:, 1:]).reshape(trg_indices.shape[0], trg_indices.shape[1])
    return src_ids, trg_ids, next_words

if __name__ == "__main__":

    print 'Testing', __name__
    train_dataset = ()
    train_x, train_y = train_dataset['x'], train_dataset['y']
    print('Number of training examples {0}'.format(len(train_x)))
