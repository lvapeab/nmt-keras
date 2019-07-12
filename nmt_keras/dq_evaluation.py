import json
import logging

import numpy as np

from keras_wrapper.extra.evaluation import get_coco_score
# warning: we modify-in-place the select dict()
# see :select:modify below
from keras_wrapper.extra.evaluation import select


def qe_metrics(pred_list, verbose, extra_vars, split, ds, set, no_ref=False):
    """
    :param pred_list: dictionary of hypothesis sentences
    :param verbose: if greater than 0 the metric measures are printed out
    :param extra_vars: extra variables
                        extra_vars['word2idx'] - dictionary mapping from words to indices
                        extra_vars['references'] - list of GT labels
    :param split: split on which we are evaluating
    :param set: DeepQuest specific
    :param no_ref: DeepQuest specific
    :return: Dictionary
    """

    final_scores = dict()


    if set=='target_text':

        final_scores = get_coco_score(pred_list[0], verbose, extra_vars, split)

    elif set=='sent_qe':
        
        sent_pred=[]
     
        if len(pred_list[0]) > 1:
            sent_pred = pred_list[0]
        else:
            sent_pred = pred_list

        if no_ref:
            final_scores = eval_sent_qe([], sent_pred, 'Sent')
        else:
            ref = eval("ds.Y_"+split+"['sent_qe']")
            final_scores = eval_sent_qe(ref, sent_pred, 'Sent')

    elif set=='doc_qe':
        
        ref = eval("ds.Y_"+split+"['doc_qe']")        
        final_scores = eval_sent_qe(ref, pred_list[0], 'Doc')


    elif set=='word_qe':
        ref = eval("ds.Y_"+split+"['word_qe']")
        final_scores = eval_word_qe(ref, pred_list[0], ds.vocabulary['word_qe'], 'Word')

    elif set=='phrase_qe':
        ref = eval("ds.Y_"+split+"['phrase_qe']")
        final_scores = eval_phrase_qe(ref, pred_list[0], ds.vocabulary['phrase_qe'], 'Phrase')

    # if verbose > 0:
    #
    #     logging.info('**Sent QE**')
    #     logging.info('Pearson %.4f' % pear_corr)
    #     logging.info('MAE %.4f' % mae)
    #     logging.info('RMSE %.4f' % rmse)
    #
    #     logging.info('**Word QE**')
    #     logging.info('Threshold %.4f' % p)
    #     logging.info('Precision %s' % precision)
    #     logging.info('Recall %s' % recall)
    #     logging.info('F-score %s' % f1)

    return final_scores


def eval_word_qe(gt_list, pred_list, vocab, qe_type):
    
    from sklearn.metrics import precision_recall_fscore_support
    from collections import defaultdict
    #print(len(pred_list))
    #print(pred_list)
    y_init = []

    for list in pred_list:
        y_init.extend(list)

    precision_eval, recall_eval, f1_eval = 0.0, 0.0, 0.0
    prec_list = []
    recall_list = []
    f1_list = []
    res_list = {}
    thresholds = np.arange(0, 1, 0.1)

    for p in thresholds:

        y_pred = []
        ref_list = []

        for i in range(len(gt_list)):

            line_ar = gt_list[i].split(' ')
            ref_list.extend(line_ar)

            for j in range(len(line_ar)):

                pred_word = y_init[i][j]
                if pred_word[vocab['words2idx']['OK']] >= p:
                    y_pred.append('OK')
                else:
                    y_pred.append('BAD')

        y_pred = np.array(y_pred)
        ref_list = np.array(ref_list)
        
        precision, recall, f1, _ = precision_recall_fscore_support(ref_list, y_pred, average=None)
        f1_list.append(np.prod(f1))
        prec_list.append(precision)
        recall_list.append(recall)

        res_list[p] = y_pred

        logging.info('**'+qe_type+'QE**')
        logging.info('Threshold %.4f' % p)
        logging.info('Precision %s' % precision)
        logging.info('Recall %s' % recall)
        logging.info('F-score %s' % f1)
        logging.info('F-score multi %s' % np.prod(f1))
        #logging.info(' '.join(y_pred))
  

    f1_list = np.array(f1_list)
    prec_list = np.array(prec_list)
    recall_list = np.array(recall_list)    

    max_f1 = np.argmax(f1_list)


    return {'precision': prec_list[max_f1],
            'recall': recall_list[max_f1],
            'f1_prod': f1_list[max_f1],
            'threshold': thresholds[max_f1],
            'pred_categorical': res_list}


def eval_phrase_qe(gt_list, pred_list, vocab, qe_type):
    
    from sklearn.metrics import precision_recall_fscore_support
    from collections import defaultdict
    #print(len(pred_list))
    #print(pred_list)
    y_init = []

    for list in pred_list:
        y_init.extend(list)

    precision_eval, recall_eval, f1_eval = 0.0, 0.0, 0.0
    prec_list = []
    recall_list = []
    f1_list = []
    res_list = {}
    thresholds = np.arange(0, 1, 0.1)

    if np.array(y_init).shape[2]==5:

        for p in thresholds:

            y_pred = []
            ref_list = []

            for i in range(len(gt_list)):

                line_ar = gt_list[i].split(' ')
                ref_list.extend(line_ar)

                for j in range(len(line_ar)):

                    pred_word = y_init[i][j]
                    if pred_word[vocab['words2idx']['BAD']] >= p:
                        y_pred.append('BAD')
                    else:
                        y_pred.append('OK')

            y_pred = np.array(y_pred)
            ref_list = np.array(ref_list)

            precision, recall, f1, _ = precision_recall_fscore_support(ref_list, y_pred, average=None)
            f1_list.append(np.prod(f1))
            prec_list.append(precision)
            recall_list.append(recall)

            res_list[p] = y_pred

            logging.info('**'+qe_type+'QE**')
            logging.info('Threshold %.4f' % p)
            logging.info('Precision %s' % precision)
            logging.info('Recall %s' % recall)
            logging.info('F-score %s' % f1)
            logging.info('F-score multi %s' % np.prod(f1))
            #logging.info(' '.join(y_pred))

    elif np.array(y_init).shape[2] == 6:

        for p1 in thresholds:

            for p2 in thresholds:

                y_pred = []
                ref_list = []

                for i in range(len(gt_list)):

                    line_ar = gt_list[i].split(' ')
                    ref_list.extend(line_ar)

                    for j in range(len(line_ar)):

                        pred_word = y_init[i][j]

                        if pred_word[vocab['words2idx']['BADWO']] >= p1:
                            y_pred.append('BADWO')
                        elif pred_word[vocab['words2idx']['BAD']] >= p2:
                            y_pred.append('BAD')
                        else:
                            y_pred.append('OK')

    f1_list = np.array(f1_list)
    prec_list = np.array(prec_list)
    recall_list = np.array(recall_list)    

    max_f1 = np.argmax(f1_list)

    return {'precision': prec_list[max_f1],
            'recall': recall_list[max_f1],
            'f1_prod': f1_list[max_f1],
            'threshold': thresholds[max_f1],
            'pred_categorical': res_list}


def eval_sent_qe(gt_list, pred_list, qe_type):

    from sklearn.metrics import mean_absolute_error, mean_squared_error
    pred_fin=[]
    for pred_batch in pred_list:
        pred_fin.extend(pred_batch.flatten())
    logging.info('**Predicted scores**')
    logging.info(pred_fin)
    if len(gt_list) > 0:
        mse = mean_squared_error(gt_list, pred_fin)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(gt_list, pred_fin)
        pear_corr = np.corrcoef(gt_list, pred_fin)[0, 1]

        logging.info('**'+qe_type+'QE**')
        logging.info('Pearson %.4f' % pear_corr)
        logging.info('MAE %.4f' % mae)
        logging.info('RMSE %.4f' % rmse)

        return {'pearson': pear_corr,
            'mae': mae,
            'rmse': rmse,
            'pred': pred_fin}
    else:
        return {'pred': pred_fin}


# :select:modify:
# Extend the `select` dict() used by keras_wrapper.extra.evaluation;
# adding in the DeepQuest specific metrics.
# Do not do this at home.
select['qe_metrics'] = qe_metrics
