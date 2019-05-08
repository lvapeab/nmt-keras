import pytest
from six import iteritems
from config import load_parameters
from utils.utils import update_parameters


def test_update_parameters():
    params = load_parameters()
    updates = {
        'ENCODER_HIDDEN_SIZE': 0,
        'BIDIRECTIONAL_ENCODER': False,
        'NEW_PARAMETER': 'new_value',
        'ADDITIONAL_OUTPUT_MERGE_MODE': 'Concat'
    }
    new_params = update_parameters(params, updates, restrict=False)
    for k, new_val in iteritems(updates):
        assert new_params[k] == updates[k]

    new_params = update_parameters(params, updates, restrict=True)
    for k, _ in iteritems(updates):
        assert new_params[k] == params.get(k, 'new_value')
    assert new_params['NEW_PARAMETER'] == updates['NEW_PARAMETER']


if __name__ == '__main__':
    pytest.main([__file__])
