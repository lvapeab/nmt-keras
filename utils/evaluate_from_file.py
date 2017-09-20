import argparse

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.ter.ter import Ter

parser = argparse.ArgumentParser(
    description="""This takes two files and a path the references (source, references),
     computes bleu, meteor, rouge and cider metrics""", formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('-t', '--hypotheses', type=str, help='Hypotheses file')
parser.add_argument('-m', '--metrics', default=['bleu', 'ter', 'meteor', 'rouge_l', 'cider'], nargs='*',
                    help='Metrics to evaluate on')
parser.add_argument('-l', '--language', type=str, default='en', help='Meteor language')
parser.add_argument('-s', '--step-size', type=int, default=0, help='Step size. 0 == Evaluate all sentences')
parser.add_argument('-r', '--references', type=argparse.FileType('r'), nargs="+",
                    help='Path to all the reference files (single-reference files)')


def load_textfiles(references, hypotheses):
    """
    Loads the references and hypothesis text files.

    :param references: Path to the references files.
    :param hypotheses: Path to the hypotheses file.
    :return:
    """
    print "The number of references is {}".format(len(references))
    hypo = {idx: [lines.strip()] for (idx, lines) in enumerate(hypotheses)}
    # take out newlines before creating dictionary
    raw_refs = [map(str.strip, r) for r in zip(*references)]
    refs = {idx: rr for idx, rr in enumerate(raw_refs)}
    # sanity check that we have the same number of references as hypothesis
    if len(hypo) != len(refs):
        raise ValueError("There is a sentence number mismatch between the inputs: \n"
                         "\t # sentences in references: %d\n"
                         "\t # sentences in hypothesis: %d" % (len(refs), len(hypo)))
    return refs, hypo


def CocoScore(ref, hyp, metrics_list=None, language='en'):
    """
    Obtains the COCO scores from the references and hypotheses.

    :param ref: Dictionary of reference sentences (id, sentence)
    :param hyp: Dictionary of hypothesis sentences (id, sentence)
    :param metrics_list: List of metrics to evaluate on
    :param language: Language of the sentences (for METEOR)
    :return: dictionary of scores
    """
    if metrics_list is None:
        metrics_list = ['bleu', 'ter', 'meteor', 'rouge_l', 'cider']

    metrics_list = map(lambda x: x.lower(), metrics_list)
    scorers = []
    if 'bleu' in metrics_list:
        scorers.append((Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]))
    if 'meteor' in metrics_list:
        scorers.append((Meteor(language), "METEOR"))
    if 'ter' in metrics_list:
        scorers.append((Ter(), "TER"))
    if 'rouge_l' in metrics_list or 'rouge' in metrics_list:
        scorers.append((Rouge(), "ROUGE_L"))
    if 'cider' in metrics_list:
        scorers.append((Cider(), "CIDEr"))

    final_scores = {}
    for scorer, method in scorers:
        score, _ = scorer.compute_score(ref, hyp)
        if isinstance(score, list):
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score
    return final_scores


if __name__ == "__main__":

    args = parser.parse_args()
    language = args.language
    hypotheses_file = open(args.hypotheses, 'r')
    step_size = args.step_size
    ref, hypothesis = load_textfiles(args.references, hypotheses_file)
    if step_size < 1:
        score = CocoScore(ref, hypothesis, metrics_list=args.metrics, language=language)
        print "Scores: "
        max_score_name_len = max([len(x) for x in score.keys()])
        for score_name in sorted(score.keys()):
            print "\t {0:{1}}".format(score_name, max_score_name_len) + ": %.5f" % score[score_name]
    else:
        n = 0
        while True:
            n += step_size
            indices = range(min(n, len(ref)))
            partial_refs = {}
            partial_hyps = {}
            for i in indices:
                partial_refs[i] = ref[i]
                partial_hyps[i] = hypothesis[i]
            score = CocoScore(partial_refs, partial_hyps, metrics_list=args.metrics, language=language)
            print str(min(n, len(ref))) + " \tScore: ", score
            if n > len(ref):
                break
