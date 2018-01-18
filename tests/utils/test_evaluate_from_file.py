import pytest
from config import load_parameters
from utils.evaluate_from_file import load_textfiles, CocoScore


def test_load_textfiles():
    params = load_parameters()
    filename = params['DATA_ROOT_PATH'] + params['TEXT_FILES']['val'] + params['TRG_LAN']
    hyp = open(filename, 'r')
    refs, hypo = load_textfiles([open(filename, 'r')], hyp)
    assert isinstance(refs, dict)
    assert isinstance(hypo, dict)
    assert refs == hypo


def test_CocoScore():
    params = load_parameters()
    filename = params['DATA_ROOT_PATH'] + params['TEXT_FILES']['val'] + params['TRG_LAN']
    hyp = open(filename, 'r')
    refs, hypo = load_textfiles([open(filename, 'r')], hyp)

    final_scores = CocoScore(refs, hypo, metrics_list=None, language=params['TRG_LAN'])
    assert isinstance(final_scores, dict)
    assert 'Bleu_1' in final_scores.keys()
    assert 'Bleu_2' in final_scores.keys()
    assert 'Bleu_3' in final_scores.keys()
    assert 'Bleu_4' in final_scores.keys()
    assert 'TER' in final_scores.keys()
    assert 'METEOR' in final_scores.keys()
    assert 'ROUGE_L' in final_scores.keys()
    assert 'CIDEr' in final_scores.keys()
    assert final_scores['Bleu_1'] - 1.0 <= 1e-6
    assert final_scores['Bleu_2'] - 1.0 <= 1e-6
    assert final_scores['Bleu_3'] - 1.0 <= 1e-6
    assert final_scores['Bleu_4'] - 1.0 <= 1e-6
    assert final_scores['TER'] - 0.0 <= 1e-6
    assert final_scores['METEOR'] - 1.0 <= 1e-6
    assert final_scores['ROUGE_L'] - 1.0 <= 1e-6
    assert final_scores['CIDEr'] - 10.0 <= 1e-1

    final_scores = CocoScore(refs, hypo, metrics_list=['BLeu'], language=params['TRG_LAN'])
    assert isinstance(final_scores, dict)
    assert 'Bleu_1' in final_scores.keys()
    assert 'Bleu_2' in final_scores.keys()
    assert 'Bleu_3' in final_scores.keys()
    assert 'Bleu_4' in final_scores.keys()
    assert 'TER' not in final_scores.keys()
    assert 'METEOR' not in final_scores.keys()
    assert 'ROUGE_L' not in final_scores.keys()
    assert 'CIDEr' not in final_scores.keys()
    assert final_scores['Bleu_1'] - 1.0 <= 1e-6
    assert final_scores['Bleu_2'] - 1.0 <= 1e-6
    assert final_scores['Bleu_3'] - 1.0 <= 1e-6
    assert final_scores['Bleu_4'] - 1.0 <= 1e-6

    final_scores = CocoScore(refs, hypo, metrics_list=['BLEU', 'ter'], language=params['TRG_LAN'])
    assert isinstance(final_scores, dict)
    assert 'Bleu_1' in final_scores.keys()
    assert 'Bleu_2' in final_scores.keys()
    assert 'Bleu_3' in final_scores.keys()
    assert 'Bleu_4' in final_scores.keys()
    assert 'TER' in final_scores.keys()
    assert 'METEOR' not in final_scores.keys()
    assert 'ROUGE_L' not in final_scores.keys()
    assert 'CIDEr' not in final_scores.keys()

    assert final_scores['Bleu_1'] - 1.0 <= 1e-6
    assert final_scores['Bleu_2'] - 1.0 <= 1e-6
    assert final_scores['Bleu_3'] - 1.0 <= 1e-6
    assert final_scores['Bleu_4'] - 1.0 <= 1e-6
    assert final_scores['TER'] - 0.0 <= 1e-6


if __name__ == '__main__':
    pytest.main([__file__])
