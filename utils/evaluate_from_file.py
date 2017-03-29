"""
Scores a file of hypothesis.
Usage:
    1. Set the references in this file (questions and annotations).
    2. python evaluate_vqa.py hypothesis.json
"""

import argparse

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.ter.ter import Ter
from pycocoevalcap.vqa import vqaEval, visual_qa

# ROOT_PATH = '/home/lvapeab/smt/tasks/image_desc/'
ROOT_PATH = '/media/HDD_2TB/DATASETS/'

questions = ROOT_PATH + '/VQA/Questions/OpenEnded_mscoco_val2014_questions.json'
annotations = ROOT_PATH + '/VQA/Annotations/mscoco_val2014_annotations.json'

parser = argparse.ArgumentParser(
    description="""This takes two files and a path the references (source, references),
     computes bleu, meteor, rouge and cider metrics""", formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('-vqa', default=False, action="store_true", help='Compute VQA metrics')
parser.add_argument('-q', type=str, default=questions, help='Path to questions file (only if the -vqa flag is active)')
parser.add_argument('-a', type=str, default=annotations,
                    help='Path to annotations file (only if the -vqa flag is active)')
parser.add_argument('-hyp', type=str, help='Hypotheses file')
parser.add_argument('-l', type=str, default='en', help='Meteor language')
parser.add_argument('-r', type=argparse.FileType('r'), nargs="+",
                    help='Path to all the reference files (single-reference files)')


def score_vqa(resFile, quesFile, annFile):
    # create vqa object and vqaRes object
    vqa_ = visual_qa.VQA(annFile, quesFile)
    vqaRes = vqa_.loadRes(resFile, quesFile)
    vqaEval_ = vqaEval.VQAEval(vqa_, vqaRes,
                               n=2)  # n is precision of accuracy (number of places after decimal), default is 2
    vqaEval_.evaluate()
    print "Overall Accuracy is: %.02f\n" % (vqaEval_.accuracy['overall'])
    return vqaEval_.accuracy['overall']


def load_textfiles(references, hypothesis):
    print "The number of references is {}".format(len(references))
    hypo = {idx: [lines.strip()] for (idx, lines) in enumerate(hypothesis)}
    # take out newlines before creating dictionary
    raw_refs = [map(str.strip, r) for r in zip(*references)]
    refs = {idx: rr for idx, rr in enumerate(raw_refs)}
    # sanity check that we have the same number of references as hypothesis
    if len(hypo) != len(refs):
        raise ValueError("There is a sentence number mismatch between the inputs: \n"
                         "\t # sentences in references: %d\n"
                         "\t # sentences in hypothesis: %d" % (len(refs), len(hypo)))
    return refs, hypo


def CocoScore(ref, hypo, language='en'):
    """
    ref, dictionary of reference sentences (id, sentence)
    hypo, dictionary of hypothesis sentences (id, sentence)
    score, dictionary of scores
    """
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(language), "METEOR"),
        (Ter(), "TER"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    final_scores = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(ref, hypo)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score
    return final_scores


if __name__ == "__main__":

    args = parser.parse_args()
    vqa_evaluation = args.vqa
    if vqa_evaluation:
        questions = args.q
        annotations = args.a
        hypotheses = args.hyp
        print "hypotheses file:", hypotheses
        score = score_vqa(hypotheses, questions, annotations)
        print "Score: ", score
    else:
        language = args.l
        hypotheses = open(args.hyp, 'r')
        ref, hypo = load_textfiles(args.r, hypotheses)
        score = CocoScore(ref, hypo, language=language)
        print "Score: ", score
