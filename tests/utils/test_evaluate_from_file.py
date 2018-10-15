import pytest
import codecs
from config import load_parameters
from utils.evaluate_from_file import load_textfiles, CocoScore


def test_load_textfiles():
    params = load_parameters()
    filename = params['DATA_ROOT_PATH'] + params['TEXT_FILES']['val'] + params['TRG_LAN']
    hyp = codecs.open(filename, 'r', encoding='utf-8')
    refs, hypo = load_textfiles([codecs.open(filename, 'r', encoding='utf-8').readlines()], hyp)
    assert isinstance(refs, dict)
    assert isinstance(hypo, dict)
    assert refs == hypo


def test_CocoScore():
    params = load_parameters()
    filename = params['DATA_ROOT_PATH'] + params['TEXT_FILES']['val'] + params['TRG_LAN']
    hyp = codecs.open(filename, 'r', encoding='utf-8')
    refs = [codecs.open(filename, 'r', encoding='utf-8').readlines()]
    refs, hypo = load_textfiles(refs, hyp)

    final_scores = CocoScore(refs, hypo, metrics_list=None, language=params['TRG_LAN'])
    assert isinstance(final_scores, dict)
    assert 'Bleu_1' in list(final_scores)
    assert 'Bleu_2' in list(final_scores)
    assert 'Bleu_3' in list(final_scores)
    assert 'Bleu_4' in list(final_scores)
    assert 'TER' in list(final_scores)
    assert 'METEOR' in list(final_scores)
    assert 'ROUGE_L' in list(final_scores)
    assert 'CIDEr' in list(final_scores)
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
    assert 'Bleu_1' in list(final_scores)
    assert 'Bleu_2' in list(final_scores)
    assert 'Bleu_3' in list(final_scores)
    assert 'Bleu_4' in list(final_scores)
    assert 'TER' not in list(final_scores)
    assert 'METEOR' not in list(final_scores)
    assert 'ROUGE_L' not in list(final_scores)
    assert 'CIDEr' not in list(final_scores)
    assert final_scores['Bleu_1'] - 1.0 <= 1e-6
    assert final_scores['Bleu_2'] - 1.0 <= 1e-6
    assert final_scores['Bleu_3'] - 1.0 <= 1e-6
    assert final_scores['Bleu_4'] - 1.0 <= 1e-6

    final_scores = CocoScore(refs, hypo, metrics_list=['BLEU', 'ter'], language=params['TRG_LAN'])
    assert isinstance(final_scores, dict)
    assert 'Bleu_1' in list(final_scores)
    assert 'Bleu_2' in list(final_scores)
    assert 'Bleu_3' in list(final_scores)
    assert 'Bleu_4' in list(final_scores)
    assert 'TER' in list(final_scores)
    assert 'METEOR' not in list(final_scores)
    assert 'ROUGE_L' not in list(final_scores)
    assert 'CIDEr' not in list(final_scores)

    assert final_scores['Bleu_1'] - 1.0 <= 1e-6
    assert final_scores['Bleu_2'] - 1.0 <= 1e-6
    assert final_scores['Bleu_3'] - 1.0 <= 1e-6
    assert final_scores['Bleu_4'] - 1.0 <= 1e-6
    assert final_scores['TER'] - 0.0 <= 1e-6

    hyp = codecs.open(filename, 'r', encoding='utf-8')
    refs = [codecs.open(filename, 'r', encoding='utf-8').readlines(),
            codecs.open(filename, 'r', encoding='utf-8').readlines(),
            codecs.open(filename, 'r', encoding='utf-8').readlines()]
    refs, hypo = load_textfiles(refs, hyp)

    final_scores = CocoScore(refs, hypo, metrics_list=None, language=params['TRG_LAN'])
    assert isinstance(final_scores, dict)
    assert 'Bleu_1' in list(final_scores)
    assert 'Bleu_2' in list(final_scores)
    assert 'Bleu_3' in list(final_scores)
    assert 'Bleu_4' in list(final_scores)
    assert 'TER' in list(final_scores)
    assert 'METEOR' in list(final_scores)
    assert 'ROUGE_L' in list(final_scores)
    assert 'CIDEr' in list(final_scores)
    assert final_scores['Bleu_1'] - 1.0 <= 1e-6
    assert final_scores['Bleu_2'] - 1.0 <= 1e-6
    assert final_scores['Bleu_3'] - 1.0 <= 1e-6
    assert final_scores['Bleu_4'] - 1.0 <= 1e-6
    assert final_scores['TER'] - 0.0 <= 1e-6
    assert final_scores['METEOR'] - 1.0 <= 1e-6
    assert final_scores['ROUGE_L'] - 1.0 <= 1e-6
    assert final_scores['CIDEr'] - 10.0 <= 1e-1

if __name__ == '__main__':
    pytest.main([__file__])
