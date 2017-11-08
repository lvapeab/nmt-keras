import unittest
from config import load_parameters
from utils.evaluate_from_file import load_textfiles, CocoScore


class TestEvaluateFromFile(unittest.TestCase):

    def test_load_textfiles(self):
        params = load_parameters()
        filename = params['DATA_ROOT_PATH'] + params['TEXT_FILES']['val'] + params['TRG_LAN']
        hyp = open(filename, 'r')
        refs, hypo = load_textfiles([open(filename, 'r')], hyp)
        self.assertIsInstance(refs, dict)
        self.assertIsInstance(hypo, dict)
        self.assertEqual(refs, hypo)

    def test_CocoScore(self):
        params = load_parameters()
        filename = params['DATA_ROOT_PATH'] + params['TEXT_FILES']['val'] + params['TRG_LAN']
        hyp = open(filename, 'r')
        refs, hypo = load_textfiles([open(filename, 'r')], hyp)

        final_scores = CocoScore(refs, hypo, metrics_list=None, language=params['TRG_LAN'])
        self.assertIsInstance(final_scores, dict)
        self.assertIn('Bleu_1', final_scores.keys())
        self.assertIn('Bleu_2', final_scores.keys())
        self.assertIn('Bleu_3', final_scores.keys())
        self.assertIn('Bleu_4', final_scores.keys())
        self.assertIn('TER', final_scores.keys())
        self.assertIn('METEOR', final_scores.keys())
        self.assertIn('ROUGE_L', final_scores.keys())
        self.assertIn('CIDEr', final_scores.keys())
        self.assertAlmostEqual(final_scores['Bleu_1'], 1.0, delta=1e-6)
        self.assertAlmostEqual(final_scores['Bleu_2'], 1.0, delta=1e-6)
        self.assertAlmostEqual(final_scores['Bleu_3'], 1.0, delta=1e-6)
        self.assertAlmostEqual(final_scores['Bleu_4'], 1.0, delta=1e-6)
        self.assertAlmostEqual(final_scores['TER'], 0.0, delta=1e-6)
        self.assertAlmostEqual(final_scores['METEOR'], 1.0, delta=1e-6)
        self.assertAlmostEqual(final_scores['ROUGE_L'], 1.0, delta=1e-6)
        self.assertAlmostEqual(final_scores['CIDEr'], 10.0, delta=1e-1)

        final_scores = CocoScore(refs, hypo, metrics_list=['BLeu'], language=params['TRG_LAN'])
        self.assertIsInstance(final_scores, dict)
        self.assertIn('Bleu_1', final_scores.keys())
        self.assertIn('Bleu_2', final_scores.keys())
        self.assertIn('Bleu_3', final_scores.keys())
        self.assertIn('Bleu_4', final_scores.keys())
        self.assertNotIn('TER', final_scores.keys())
        self.assertNotIn('METEOR', final_scores.keys())
        self.assertNotIn('ROUGE_L', final_scores.keys())
        self.assertNotIn('CIDEr', final_scores.keys())
        self.assertAlmostEqual(final_scores['Bleu_1'], 1.0, delta=1e-6)
        self.assertAlmostEqual(final_scores['Bleu_2'], 1.0, delta=1e-6)
        self.assertAlmostEqual(final_scores['Bleu_3'], 1.0, delta=1e-6)
        self.assertAlmostEqual(final_scores['Bleu_4'], 1.0, delta=1e-6)

        final_scores = CocoScore(refs, hypo, metrics_list=['BLEU', 'ter'], language=params['TRG_LAN'])
        self.assertIsInstance(final_scores, dict)
        self.assertIn('Bleu_1', final_scores.keys())
        self.assertIn('Bleu_2', final_scores.keys())
        self.assertIn('Bleu_3', final_scores.keys())
        self.assertIn('Bleu_4', final_scores.keys())
        self.assertIn('TER', final_scores.keys())
        self.assertNotIn('METEOR', final_scores.keys())
        self.assertNotIn('ROUGE_L', final_scores.keys())
        self.assertNotIn('CIDEr', final_scores.keys())
        self.assertAlmostEqual(final_scores['Bleu_1'], 1.0, delta=1e-6)
        self.assertAlmostEqual(final_scores['Bleu_2'], 1.0, delta=1e-6)
        self.assertAlmostEqual(final_scores['Bleu_3'], 1.0, delta=1e-6)
        self.assertAlmostEqual(final_scores['Bleu_4'], 1.0, delta=1e-6)
        self.assertAlmostEqual(final_scores['TER'], 0.0, delta=1e-6)
if __name__ == '__main__':
    unittest.main()
