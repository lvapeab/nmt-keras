import json
import logging
import numpy as np
from sklearn import metrics as sklearn_metrics

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge

logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(message)s', datefmt='%d/%m/%Y %H:%M:%S')


def get_coco_score(pred_list, verbose, extra_vars, split):
    """
    COCO challenge metrics

    :param pred_list: dictionary of hypothesis sentences (id, sentence)
    :param verbose: f greater than 0 the metric measures are printed out
    :param extra_vars: extra variables, here are:
            extra_vars['references'] - mapping from sample indices to a list with all valid captions (id, [sentences])
            extra_vars['tokenize_f'] - tokenization function used during model training (used again for validation)
    :param split: Split on which to compute the metrics
    :return:
    """
    gts = extra_vars[split]['references']
    hypo = {idx: map(extra_vars['tokenize_f'], [lines.strip()]) for (idx, lines) in enumerate(pred_list)}
    refs = {idx: map(extra_vars['tokenize_f'], gts[idx]) for idx in gts.keys()}

    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(language=extra_vars['language']), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]

    final_scores = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(refs, hypo)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score

    if verbose > 0:
        logging.info('Computing coco scores on the %s split...' % split)
        for metric in sorted(final_scores):
            value = final_scores[metric]
            logging.info(metric + ': ' + str(value))

    return final_scores


def eval_multiclass_metrics(pred_list, verbose, extra_vars, split):
    """
    Multiclass classification metrics
    see multilabel ranking metrics in sklearn library for more info:
        http://scikit-learn.org/stable/modules/model_evaluation.html#multilabel-ranking-metrics
        
    # Arguments
        gt_list, dictionary of reference sentences
        pred_list, dictionary of hypothesis sentences
        verbose - if greater than 0 the metric measures are printed out
        extra_vars - extra variables, here are:
                extra_vars['word2idx'] - dictionary mapping from words to indices
                extra_vars['references'] - list of GT labels
    """
    word2idx = extra_vars[split]['word2idx']
    n_classes = len(word2idx)
    n_samples = len(pred_list)
    
    # Create prediction matrix
    y_pred = np.zeros((n_samples, n_classes))
    for i_s, sample in enumerate(pred_list):
        for word in sample:
            y_pred[i_s, word2idx[word]] = 1

    gt_list = extra_vars[split]['references']
    y_gt = np.array(gt_list)

    # Compute Coverage Error
    coverr = sklearn_metrics.coverage_error(y_gt, y_pred)
    # Compute Label Ranking AvgPrec
    avgprec = sklearn_metrics.label_ranking_average_precision_score(y_gt, y_pred)
    # Compute Label Ranking Loss
    rankloss = sklearn_metrics.label_ranking_loss(y_gt, y_pred)

    if verbose > 0:
        logging.info('Coverage Error (best: avg labels per sample = %f): %f' %(np.sum(y_gt)/float(n_samples), coverr))
        logging.info('Label Ranking Average Precision (best: 1.0): %f' % avgprec)
        logging.info('Label Ranking Loss (best: 0.0): %f' % rankloss)
    
    return {'coverage error': coverr,
            'average precision': avgprec,
            'ranking loss': rankloss}


def multilabel_metrics(pred_list, verbose, extra_vars, split):
    """
    Multiclass classification metrics
    see multilabel ranking metrics in sklearn library for more info:
    http://scikit-learn.org/stable/modules/model_evaluation.html#multilabel-ranking-metrics

    :param pred_list: dictionary of hypothesis sentences
    :param verbose:
    :param extra_vars: extra variables, here are:
                extra_vars['word2idx'] - dictionary mapping from words to indices
    :param split:
    :return:
    """

    n_classes = extra_vars['n_classes']
    n_samples = len(pred_list)
    gt_list = extra_vars[split]['references']
    pred_class_list = [np.argmax(sample_score) for sample_score in pred_list]
    # Create prediction matrix
    y_pred = np.zeros((n_samples, n_classes))
    y_gt = np.zeros((n_samples, n_classes))
    for i_s, pred_class in enumerate(pred_class_list):
        y_pred[i_s, pred_class] = 1
    try:
        values_gt = gt_list.values()
    except ValueError:
        values_gt = gt_list
    for i_s, gt_class in enumerate(values_gt):
        y_gt[i_s, gt_class] = 1

    # Compute Coverage Error
    accuracy = sklearn_metrics.accuracy_score(y_gt, y_pred)
    if verbose > 0:
        logging.info('Accuracy: %f' %
                     accuracy)

    return {'accuracy': accuracy}

# EVALUATION FUNCTIONS SELECTOR
# List of evaluation functions and their identifiers (will be used in parameters['METRICS'])
select = {
         'coco': get_coco_score,                 # MS COCO evaluation library (BLEU, METEOR and CIDEr scores)
         'multiclass': eval_multiclass_metrics,  # Set of multiclass classification metrics from sklearn
         'multilabel_metrics': multilabel_metrics,  # Set of multilabel classification metrics from sklearn
         }
