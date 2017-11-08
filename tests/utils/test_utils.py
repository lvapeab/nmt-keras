import unittest
from config import load_parameters
from utils.utils import update_parameters


class TestUtils(unittest.TestCase):
    def test_update_parameters(self):
        params = load_parameters()
        updates = {
        'ENCODER_HIDDEN_SIZE': 0,
        'BIDIRECTIONAL_ENCODER': False,
        'NEW_PARAMETER': 'new_value',
        'ADDITIONAL_OUTPUT_MERGE_MODE': 'Concat'
        }
        new_params = update_parameters(params, updates, restrict=False)
        for k, new_val in updates.iteritems():
            self.assertEqual(new_params[k], updates[k])

        new_params = update_parameters(params, updates, restrict=True)
        for k, _ in updates.iteritems():
            self.assertEqual(new_params[k], params.get(k, 'new_value'))
        self.assertEqual(new_params['NEW_PARAMETER'], updates['NEW_PARAMETER'])

if __name__ == '__main__':
    unittest.main()
