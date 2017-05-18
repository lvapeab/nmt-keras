import unittest
from config import load_parameters
from data_engine.prepare_data import build_dataset
from keras_wrapper.dataset import Dataset
from keras_wrapper.cnn_model import Model_Wrapper
from model_zoo import TranslationModel

class TestNMTConstruction(unittest.TestCase):

    def test_build_datset(self):
        params = load_parameters()
        params['REBUILD_DATASET'] = True
        dataset = build_dataset(params)
        params['INPUT_VOCABULARY_SIZE'] = dataset.vocabulary_len[params['INPUTS_IDS_DATASET'][0]]
        params['OUTPUT_VOCABULARY_SIZE'] = dataset.vocabulary_len[params['OUTPUTS_IDS_DATASET'][0]]
        for rnn_type in ['LSTM', 'GRU']:
            for n_layers in range(2):
                params['N_LAYERS_DECODER'] = n_layers
                params['N_LAYERS_ENCODER'] = n_layers
                params['RNN_TYPE'] = rnn_type
                nmt_model = TranslationModel(params, model_type=params['MODEL_TYPE'],
                                             verbose=params['VERBOSE'],
                                             model_name=params['MODEL_NAME'],
                                             vocabularies=dataset.vocabulary,
                                             store_path=params['STORE_PATH'])
                self.assertIsInstance(nmt_model, Model_Wrapper)

        # Check Inputs
        inputMapping = dict()
        for i, id_in in enumerate(params['INPUTS_IDS_DATASET']):
            pos_source = dataset.ids_inputs.index(id_in)
            id_dest = nmt_model.ids_inputs[i]
            inputMapping[id_dest] = pos_source
        nmt_model.setInputsMapping(inputMapping)
        outputMapping = dict()
        for i, id_out in enumerate(params['OUTPUTS_IDS_DATASET']):
            pos_target = dataset.ids_outputs.index(id_out)
            id_dest = nmt_model.ids_outputs[i]
            outputMapping[id_dest] = pos_target
        nmt_model.setOutputsMapping(outputMapping)


if __name__ == '__main__':
    unittest.main()