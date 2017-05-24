import argparse

# Scores a file of hypothesis.
# Usage:
#     1. Set the references in this file.
#     2. python evaluate_from_file.py -hyp hypothesis -r references

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.ter.ter import Ter

parser = argparse.ArgumentParser(
    description="""This takes two files and a
     path to the references (source, references),
     and computes bleu, meteor,
     rouge, cider and TER metrics""",
    formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('-t', '--hypotheses', type=str,
                    help='Hypotheses file')
parser.add_argument('-l', '--language', type=str,
                    default='en',
                    help='Meteor language')
parser.add_argument('-s', '--step-size',
                    type=int, default=0,
                    help='Step size. 0 == Evaluate all sentences')
parser.add_argument('-r', '--references',
                    type=argparse.FileType('r'),
                    nargs="+",
                    help='Path to all the '
                         'reference files (single-reference files)')


def load_textfiles(references, hypothesis):
    """
    Loads the references and hypothesis text files.

    :param references: Path to the references files.
    :param hypothesis: Path to the hypotheses file.
    :return:
    """
    print "The number of references is {}".format(len(references))
    hypo = {idx: [lines.strip()] for (idx, lines) in enumerate(hypothesis)}
    # take out newlines before creating dictionary
    raw_refs = [map(str.strip, r) for r in zip(*references)]
    refs = {idx: rr for idx, rr in enumerate(raw_refs)}
    # sanity check that we have the same number of references as hypothesis
    if len(hypo) != len(refs):
        raise ValueError("There is a sentence number mismatch between"
                         " the inputs: \n"
                         "\t # sentences in references: %d\n"
                         "\t # sentences in hypothesis: %d" %
                         (len(refs), len(hypo)))
    return refs, hypo


def CocoScore(ref, hypo, language='en'):
    """
    Obtains the COCO scores from the references and hypotheses.

    :param ref: Dictionary of reference sentences (id, sentence)
    :param hypo: Dictionary of hypothesis sentences (id, sentence)
    :param language: Language of the sentences (for METEOR)
    :return: dictionary of scores
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
        score, _ = scorer.compute_score(ref, hypo)
        if isinstance(score, list):
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score
    return final_scores


if __name__ == "__main__":

    args = parser.parse_args()
    language = args.language
    hypotheses = open(args.hypotheses, 'r')
    step_size = args.step_size
    ref, hypo = load_textfiles(args.references, hypotheses)
    if step_size < 1:
        score = CocoScore(ref, hypo, language=language)
        print "Scores: "
        max_score_name_len = max([len(x) for x in score.keys()])
        for score_name in sorted(score.keys()):
            print "\t {0:{1}}".format(score_name, max_score_name_len) \
                  + ": %.5f" % score[score_name]
    else:
        n = 0
        while True:
            n += step_size
            indices = range(min(n, len(ref)))
            partial_refs = {}
            partial_hyps = {}
            for i in indices:
                partial_refs[i] = ref[i]
                partial_hyps[i] = hypo[i]
            score = CocoScore(partial_refs, partial_hyps, language=language)
            print str(min(n, len(ref))) + " \tScore: ", score
            if n > len(ref):
                break
