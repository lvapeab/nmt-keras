# -*- coding: utf-8 -*-
from keras_wrapper.extra.callbacks import *


def buildCallbacks(params, model, dataset):
    """
    Builds the selected set of callbacks run during the training of the model:
        * PrintPerformanceMetricOnEpochEndOrEachNUpdates: Evaluates the model in the validation set given a number of epochs/updates.
        * SampleEachNUpdates: Shows several translation samples during training.


    :param dict params: Dictionary of network hyperparameters.
    :param Model_Wrapper model: Model instance on which to apply the callback.
    :param Dataset dataset: Dataset instance on which to apply the callback.
    :return: list of callbacks to pass to the Keras' training.
    """

    callbacks = []

    if params['METRICS'] or params['SAMPLE_ON_SETS']:
        # Evaluate training
        extra_vars = {'language': params.get('TRG_LAN', 'en'),
                      'n_parallel_loaders': params['PARALLEL_LOADERS'],
                      'tokenize_f': eval('dataset.' + params.get('TOKENIZATION_METHOD', 'tokenize_none')),
                      'detokenize_f': eval('dataset.' + params.get('DETOKENIZATION_METHOD', 'detokenize_none')),
                      'apply_detokenization': params.get('APPLY_DETOKENIZATION', False),
                      'tokenize_hypotheses': params.get('TOKENIZE_HYPOTHESES', True),
                      'tokenize_references': params.get('TOKENIZE_REFERENCES', True)
                      }

        input_text_id = params['INPUTS_IDS_DATASET'][0]
        vocab_x = dataset.vocabulary[input_text_id]['idx2words']
        vocab_y = dataset.vocabulary[params['OUTPUTS_IDS_DATASET'][0]]['idx2words']
        if params['BEAM_SEARCH']:
            extra_vars['beam_size'] = params.get('BEAM_SIZE', 6)
            extra_vars['state_below_index'] = params.get('BEAM_SEARCH_COND_INPUT', -1)
            extra_vars['maxlen'] = params.get('MAX_OUTPUT_TEXT_LEN_TEST', 30)
            extra_vars['optimized_search'] = params.get('OPTIMIZED_SEARCH', True)
            extra_vars['model_inputs'] = params['INPUTS_IDS_MODEL']
            extra_vars['model_outputs'] = params['OUTPUTS_IDS_MODEL']
            extra_vars['dataset_inputs'] = params['INPUTS_IDS_DATASET']
            extra_vars['dataset_outputs'] = params['OUTPUTS_IDS_DATASET']
            extra_vars['search_pruning'] = params.get('SEARCH_PRUNING', False)
            extra_vars['normalize_probs'] = params.get('NORMALIZE_SAMPLING', False)
            extra_vars['alpha_factor'] = params.get('ALPHA_FACTOR', 1.)
            extra_vars['coverage_penalty'] = params.get('COVERAGE_PENALTY', False)
            extra_vars['length_penalty'] = params.get('LENGTH_PENALTY', False)
            extra_vars['length_norm_factor'] = params.get('LENGTH_NORM_FACTOR', 0.0)
            extra_vars['coverage_norm_factor'] = params.get('COVERAGE_NORM_FACTOR', 0.0)
            extra_vars['state_below_maxlen'] = -1 if params.get('PAD_ON_BATCH', True) \
                else params.get('MAX_OUTPUT_TEXT_LEN', 50)
            extra_vars['pos_unk'] = params['POS_UNK']
            extra_vars['output_max_length_depending_on_x'] = params.get('MAXLEN_GIVEN_X', True)
            extra_vars['output_max_length_depending_on_x_factor'] = params.get('MAXLEN_GIVEN_X_FACTOR', 3)
            extra_vars['output_min_length_depending_on_x'] = params.get('MINLEN_GIVEN_X', True)
            extra_vars['output_min_length_depending_on_x_factor'] = params.get('MINLEN_GIVEN_X_FACTOR', 2)
            extra_vars['attend_on_output'] = params.get('ATTEND_ON_OUTPUT', 'transformer' in params['MODEL_TYPE'].lower())
            extra_vars['glossary'] = None if params.get('GLOSSARY', None) is None else pkl2dict(params.get('GLOSSARY'))

            if params['POS_UNK']:
                extra_vars['heuristic'] = params['HEURISTIC']
                if params['HEURISTIC'] > 0:
                    extra_vars['mapping'] = dataset.mapping

        if params['METRICS']:
            for s in params['EVAL_ON_SETS']:
                extra_vars[s] = dict()
                extra_vars[s]['references'] = dataset.extra_variables[s][params['OUTPUTS_IDS_DATASET'][0]]
            callback_metric = PrintPerformanceMetricOnEpochEndOrEachNUpdates(model,
                                                                             dataset,
                                                                             gt_id=params['OUTPUTS_IDS_DATASET'][0],
                                                                             metric_name=params['METRICS'],
                                                                             set_name=params['EVAL_ON_SETS'],
                                                                             batch_size=params['BATCH_SIZE'],
                                                                             each_n_epochs=params['EVAL_EACH'],
                                                                             extra_vars=extra_vars,
                                                                             reload_epoch=params['RELOAD'],
                                                                             is_text=True,
                                                                             input_text_id=input_text_id,
                                                                             index2word_y=vocab_y,
                                                                             index2word_x=vocab_x,
                                                                             sampling_type=params['SAMPLING'],
                                                                             beam_search=params['BEAM_SEARCH'],
                                                                             save_path=model.model_path,
                                                                             start_eval_on_epoch=params['START_EVAL_ON_EPOCH'],
                                                                             write_samples=True,
                                                                             write_type=params['SAMPLING_SAVE_MODE'],
                                                                             eval_on_epochs=params['EVAL_EACH_EPOCHS'],
                                                                             save_each_evaluation=params['SAVE_EACH_EVALUATION'],
                                                                             do_plot=params.get('PLOT_EVALUATION', False),
                                                                             verbose=params['VERBOSE'])

            callbacks.append(callback_metric)

        if params['SAMPLE_ON_SETS']:
            callback_sampling = SampleEachNUpdates(model,
                                                   dataset,
                                                   gt_id=params['OUTPUTS_IDS_DATASET'][0],
                                                   set_name=params['SAMPLE_ON_SETS'],
                                                   n_samples=params['N_SAMPLES'],
                                                   each_n_updates=params['SAMPLE_EACH_UPDATES'],
                                                   extra_vars=extra_vars,
                                                   reload_epoch=params['RELOAD'],
                                                   batch_size=params['BATCH_SIZE'],
                                                   is_text=True,
                                                   index2word_x=vocab_x,
                                                   index2word_y=vocab_y,
                                                   print_sources=True,
                                                   in_pred_idx=params['INPUTS_IDS_DATASET'][0],
                                                   sampling_type=params['SAMPLING'],
                                                   beam_search=params['BEAM_SEARCH'],
                                                   start_sampling_on_epoch=params['START_SAMPLING_ON_EPOCH'],
                                                   verbose=params['VERBOSE'])
            callbacks.append(callback_sampling)
    return callbacks
