import unittest
import theano
from config import load_parameters
from data_engine.prepare_data import build_dataset
from keras_wrapper.cnn_model import Model_Wrapper, loadModel
from keras_wrapper.extra.callbacks import PrintPerformanceMetricOnEpochEndOrEachNUpdates
from main import buildCallbacks
from model_zoo import TranslationModel


class TestNMT(unittest.TestCase):

    def test_build(self):
        params = load_parameters()
        params['DATASET_STORE_PATH'] = './'
        params['REBUILD_DATASET'] = True
        dataset = build_dataset(params)
        params['INPUT_VOCABULARY_SIZE'] = \
            dataset.vocabulary_len[params['INPUTS_IDS_DATASET'][0]]
        params['OUTPUT_VOCABULARY_SIZE'] = \
            dataset.vocabulary_len[params['OUTPUTS_IDS_DATASET'][0]]
        for rnn_type in ['LSTM', 'GRU']:
            for n_layers in range(2):
                params['N_LAYERS_DECODER'] = n_layers
                params['N_LAYERS_ENCODER'] = n_layers
                params['RNN_TYPE'] = rnn_type
                nmt_model = \
                    TranslationModel(params,
                                     model_type=params['MODEL_TYPE'],
                                     verbose=params['VERBOSE'],
                                     model_name=params['MODEL_NAME'],
                                     vocabularies=dataset.vocabulary,
                                     store_path=params['STORE_PATH'],
                                     clear_dirs=False)
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
        return True

    @classmethod
    def test_train_and_load(self):
        if theano.config.device == 'gpu':
            def test_train():
                params = load_parameters()
                params['REBUILD_DATASET'] = True
                params['DATASET_STORE_PATH'] = './'
                dataset = build_dataset(params)
                params['INPUT_VOCABULARY_SIZE'] = \
                    dataset.vocabulary_len[params['INPUTS_IDS_DATASET'][0]]
                params['OUTPUT_VOCABULARY_SIZE'] = \
                    dataset.vocabulary_len[params['OUTPUTS_IDS_DATASET'][0]]

                params['SOURCE_TEXT_EMBEDDING_SIZE'] = 2
                params['TARGET_TEXT_EMBEDDING_SIZE'] = 2
                params['ENCODER_HIDDEN_SIZE'] = 2
                params['DECODER_HIDDEN_SIZE'] = 2
                params['ATTENTION_SIZE'] = 2
                params['SKIP_VECTORS_HIDDEN_SIZE'] = 2
                params['DEEP_OUTPUT_LAYERS'] = [('linear', 2)]
                params['STORE_PATH'] = './'
                nmt_model = \
                    TranslationModel(params,
                                     model_type=params['MODEL_TYPE'],
                                     verbose=params['VERBOSE'],
                                     model_name=params['MODEL_NAME'],
                                     vocabularies=dataset.vocabulary,
                                     store_path=params['STORE_PATH'],
                                     clear_dirs=False)

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
                callbacks = buildCallbacks(params, nmt_model, dataset)
                training_params = {'n_epochs': 1,
                                   'batch_size': 50,
                                   'homogeneous_batches': False,
                                   'maxlen': 10,
                                   'joint_batches': params['JOINT_BATCHES'],
                                   'lr_decay': params['LR_DECAY'],
                                   'lr_gamma': params['LR_GAMMA'],
                                   'epochs_for_save': 1,
                                   'verbose': params['VERBOSE'],
                                   'eval_on_sets': params['EVAL_ON_SETS_KERAS'],
                                   'n_parallel_loaders': params['PARALLEL_LOADERS'],
                                   'extra_callbacks': callbacks,
                                   'reload_epoch': 0,
                                   'epoch_offset': 0,
                                   'data_augmentation': False,
                                   'patience': 1,  # early stopping parameters
                                   'metric_check': 'Bleu_4',
                                   'eval_on_epochs': True,
                                   'each_n_epochs': 1,
                                   'start_eval_on_epoch': 0}
                nmt_model.trainNet(dataset, training_params)
                return True

            test_train()
            params = load_parameters()
            params['REBUILD_DATASET'] = True
            params['DATASET_STORE_PATH'] = './'
            dataset = build_dataset(params)
            params['INPUT_VOCABULARY_SIZE'] = \
                dataset.vocabulary_len[params['INPUTS_IDS_DATASET'][0]]
            params['OUTPUT_VOCABULARY_SIZE'] = \
                dataset.vocabulary_len[params['OUTPUTS_IDS_DATASET'][0]]

            # Load model
            nmt_model = loadModel('./', 1, reload_epoch=True)
            nmt_model.setOptimizer()

            for s in ['val']:

                # Evaluate training
                extra_vars = {'language': params.get('TRG_LAN', 'en'),
                              'n_parallel_loaders': params['PARALLEL_LOADERS'],
                              'tokenize_f': eval('dataset.' + params['TOKENIZATION_METHOD']),
                              'detokenize_f': eval('dataset.' + params['DETOKENIZATION_METHOD']),
                              'apply_detokenization': params['APPLY_DETOKENIZATION'],
                              'tokenize_hypotheses': params['TOKENIZE_HYPOTHESES'],
                              'tokenize_references': params['TOKENIZE_REFERENCES']}
                vocab = dataset.vocabulary[params['OUTPUTS_IDS_DATASET'][0]]['idx2words']
                extra_vars[s] = dict()
                extra_vars[s]['references'] = dataset.extra_variables[s][params['OUTPUTS_IDS_DATASET'][0]]
                input_text_id = None
                vocab_src = None
                if params['BEAM_SIZE']:
                    extra_vars['beam_size'] = params.get('BEAM_SIZE', 6)
                    extra_vars['state_below_index'] = params.get('BEAM_SEARCH_COND_INPUT', -1)
                    extra_vars['maxlen'] = params.get('MAX_OUTPUT_TEXT_LEN_TEST', 30)
                    extra_vars['optimized_search'] = params.get('OPTIMIZED_SEARCH', True)
                    extra_vars['model_inputs'] = params['INPUTS_IDS_MODEL']
                    extra_vars['model_outputs'] = params['OUTPUTS_IDS_MODEL']
                    extra_vars['dataset_inputs'] = params['INPUTS_IDS_DATASET']
                    extra_vars['dataset_outputs'] = params['OUTPUTS_IDS_DATASET']
                    extra_vars['normalize_probs'] = params.get('NORMALIZE_SAMPLING', False)
                    extra_vars['alpha_factor'] = params.get('ALPHA_FACTOR', 1.0)
                    extra_vars['coverage_penalty'] = params.get('COVERAGE_PENALTY', False)
                    extra_vars['length_penalty'] = params.get('LENGTH_PENALTY', False)
                    extra_vars['length_norm_factor'] = params.get('LENGTH_NORM_FACTOR', 0.0)
                    extra_vars['coverage_norm_factor'] = params.get('COVERAGE_NORM_FACTOR', 0.0)
                    extra_vars['pos_unk'] = params['POS_UNK']
                    if params['POS_UNK']:
                        extra_vars['heuristic'] = params['HEURISTIC']
                        input_text_id = params['INPUTS_IDS_DATASET'][0]
                        vocab_src = dataset.vocabulary[input_text_id]['idx2words']
                        if params['HEURISTIC'] > 0:
                            extra_vars['mapping'] = dataset.mapping

                callback_metric = PrintPerformanceMetricOnEpochEndOrEachNUpdates(nmt_model,
                                                                                 dataset,
                                                                                 gt_id=params['OUTPUTS_IDS_DATASET'][0],
                                                                                 metric_name=params['METRICS'],
                                                                                 set_name=params['EVAL_ON_SETS'],
                                                                                 batch_size=params['BATCH_SIZE'],
                                                                                 each_n_epochs=params['EVAL_EACH'],
                                                                                 extra_vars=extra_vars,
                                                                                 reload_epoch=1,
                                                                                 is_text=True,
                                                                                 input_text_id=input_text_id,
                                                                                 save_path=nmt_model.model_path,
                                                                                 index2word_y=vocab,
                                                                                 index2word_x=vocab_src,
                                                                                 sampling_type=params['SAMPLING'],
                                                                                 beam_search=params['BEAM_SEARCH'],
                                                                                 start_eval_on_epoch=0,
                                                                                 write_samples=True,
                                                                                 write_type=params['SAMPLING_SAVE_MODE'],
                                                                                 eval_on_epochs=params['EVAL_EACH_EPOCHS'],
                                                                                 save_each_evaluation=False,
                                                                                 verbose=params['VERBOSE'])

                callback_metric.evaluate(1, counter_name='epoch' if params['EVAL_EACH_EPOCHS'] else 'update')
                return True
        else:
            pass

if __name__ == '__main__':
    unittest.main()