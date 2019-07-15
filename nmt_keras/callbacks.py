from __future__ import print_function

import warnings

from keras.callbacks import Callback as KerasCallback
from keras_wrapper.utils import decode_predictions_one_hot, decode_predictions_beam_search, decode_predictions, \
    decode_multilabel
from keras_wrapper.extra.read_write import *

import keras_wrapper.extra.evaluation as evaluation

from . import dq_evaluation


def checkDefaultParamsBeamSearch(params):
    required_params = ['model_inputs', 'model_outputs', 'dataset_inputs', 'dataset_outputs']
    default_params = {'max_batch_size': 50,
                      'beam_size': 5,
                      'maxlen': 30,
                      'normalize': False,
                      'words_so_far': False,
                      'n_parallel_loaders': 5,
                      'optimized_search': False,
                      'temporally_linked': False,
                      'link_index_id': 'link_index',
                      'state_below_index': -1,
                      'pos_unk': False,
                      'max_eval_samples': None,
                      'search_pruning': False,
                      'normalize_probs': False,
                      'alpha_factor': 0.0,
                      'coverage_penalty': False,
                      'length_penalty': False,
                      'length_norm_factor': 0.0,
                      'coverage_norm_factor': 0.0,
                      'output_max_length_depending_on_x': False,
                      'output_max_length_depending_on_x_factor': 3,
                      'output_min_length_depending_on_x': False,
                      'output_min_length_depending_on_x_factor': 2
                      }

    for k, v in params.iteritems():
        if k in default_params.keys() or k in required_params:
            default_params[k] = v

    for k in required_params:
        if k not in default_params:
            raise Exception('The beam search parameter ' + k + ' must be specified.')

    return default_params


###################################################
# Performance evaluation callbacks
###################################################

class EvalPerformance(KerasCallback):
    def __init__(self,
                 model,
                 dataset,
                 gt_id,
                 metric_name,
                 set_name,
                 batch_size,
                 gt_pos=None,
                 each_n_epochs=1,
                 max_eval_samples=None,
                 extra_vars=None,
                 normalize=False,
                 output_types=None,
                 is_text=False,
                 is_multilabel=False,
                 multilabel_idx=None,
                 min_pred_multilabel=0.5,
                 index2word_y=None,
                 input_text_id=None,
                 input_id=None,
                 index2word_x=None,
                 sampling='max_likelihood',
                 beam_search=False,
                 beam_batch_size=None,
                 write_samples=False,
                 write_type='list',
                 save_path='logs/performance.',
                 reload_epoch=0,
                 eval_on_epochs=True,
                 start_eval_on_epoch=0,
                 is_3DLabel=False,
                 sampling_type='max_likelihood',
                 save_each_evaluation=True,
                 out_pred_idx=None,
                 max_plot=1.0,
                 do_plot=True,
                 verbose=1,
                 no_ref=False):
        """
        Evaluates a model each N epochs or updates

        :param model: model to evaluate
        :param dataset: instance of the class Dataset in keras_wrapper.dataset

        :param gt_id: identifier in the Dataset instance of the output data to evaluate
        :param gt_pos: position of the GT output to evaluate in model's outputs

        :param metric_name: name of the performance metric
        :param set_name: list with the names of the set splits that will be evaluated
        :param batch_size: batch size used during sampling
        :param each_n_epochs: sampling each this number of epochs or updates
        :param max_eval_samples: maximum number of samples evaluated
        :param extra_vars: dictionary of extra variables. See evaluation metrics in keras_wrapper/extra/evaluation.py
                           for assigning the needed extra variables.
        :param output_types: list with type identifiers of the different outputs to evaluate (len must coincide with gt_post)
        :param normalize: switch on/off data normalization
        :param min_pred_multilabel: minimum prediction value considered for positive prediction
        :param index2word_y: mapping from the indices to words (only needed if is_text==True)
        :param input_text_id:
        :param input_id: identifier in the Dataset instance of the input data
        :param index2word_x: mapping from the indices to words (only needed if is_text==True)
        :param sampling: sampling mechanism used (only used if is_text==True)
        :param beam_search: whether to use a beam search method or not
        :param beam_batch_size: batch size allowed during beam search
        :param write_samples: flag for indicating if we want to write the predicted data in a file (text or image)
        :param write_type: type of data used for writing predictions
        :param save_path: path to dumb the logs
        :param reload_epoch: reloading epoch
        :param eval_on_epochs: eval each epochs (True) or each updates (False)
        :param start_eval_on_epoch: only starts evaluating model if a given epoch has been reached
        :param is_3DLabel: defines if the predicted info is of type 3DLabels
        :param sampling_type: type of sampling used (multinomial or max_likelihood)
        :param save_each_evaluation: save the model each time we evaluate (epochs or updates)
        :param out_pred_idx: index of the output prediction used for evaluation
                             (only applicable if model has more than one output, else set to None)
        :param max_plot: maximum value shown on the performance plots generated
        :param verbose: verbosity level; by default 1
        :param do_plot: plot results so far
        :param no_ref: in case when testing on unlabeled test set 


        Deprecated outputs

        :param is_text: defines if the predicted info is of type text (in that case the data will be
                        converted from values into a textual representation)
        :param is_multilabel: are we applying multi-label prediction?
        :param multilabel_idx: output index where to apply the evaluation (set to None if the model has a single output)

        """
        if gt_pos is None:
            gt_pos = []
        if extra_vars is None:
            extra_vars = dict()

        if type(gt_id) is not list:
            gt_id = [gt_id]

        self.model_to_eval = model
        self.ds = dataset

        self.gt_id = gt_id
        self.gt_pos = gt_pos

        self.input_text_id = input_text_id
        self.input_id = input_id
        self.index2word_x = index2word_x
        self.index2word_y = index2word_y

        # Deprecated
        self.is_text = is_text
        self.is_multilabel = is_multilabel
        self.multilabel_idx = multilabel_idx
        # Use instead
        self.output_types = output_types

        self.min_pred_multilabel = min_pred_multilabel
        self.is_3DLabel = is_3DLabel
        self.sampling = sampling
        self.beam_search = beam_search
        self.beam_batch_size = beam_batch_size
        self.metric_name = metric_name
        self.set_name = set_name
        self.max_eval_samples = max_eval_samples
        self.batch_size = batch_size
        self.each_n_epochs = each_n_epochs
        self.extra_vars = extra_vars
        self.normalize = normalize
        self.save_path = save_path
        self.eval_on_epochs = eval_on_epochs
        self.start_eval_on_epoch = start_eval_on_epoch
        self.write_type = write_type
        self.sampling_type = sampling_type
        self.write_samples = write_samples
        self.out_pred_idx = out_pred_idx
        self.best_score = -1
        self.best_epoch = -1
        self.wait = 0
        self.verbose = verbose
        self.cum_update = 0
        self.epoch = reload_epoch
        self.max_plot = max_plot
        self.save_each_evaluation = save_each_evaluation
        self.written_header = False
        self.do_plot = do_plot
        self.no_ref = no_ref
        create_dir_if_not_exists(self.save_path)

        # Single-output model
        if not self.gt_pos or self.gt_pos == 0:
            self.metric_name = [self.metric_name]
            self.write_type = [self.write_type]
            self.index2word_y = [self.index2word_y]
            self.index2word_x = [self.index2word_x]
            if 0 not in self.extra_vars.keys():
                self.extra_vars[0] = self.extra_vars

            if self.output_types is None:
                if self.is_multilabel:
                    self.output_types = ['binary']
                elif self.is_text:
                    self.output_types = ['text']
            else:
                self.output_types = [self.output_types]


        super(EvalPerformance, self).__init__()

    def on_epoch_end(self, epoch, logs={}):
        """
        On epoch end, sample and evaluate on the specified datasets.
        :param epoch: Current epoch number
        :param logs:
        :return:
        """
        epoch += 1  # start by index 1
        self.epoch = epoch
        if not self.eval_on_epochs:
            return
        if epoch < self.start_eval_on_epoch:
            if self.verbose > 0:
                logging.info('Not evaluating until end of epoch ' + str(self.start_eval_on_epoch))
            return
        elif (epoch - self.start_eval_on_epoch) % self.each_n_epochs != 0:
            if self.verbose > 0:
                logging.info('Evaluating only every ' + str(self.each_n_epochs) + ' epochs')
            return
        self.evaluate(epoch, counter_name='epoch')

    def on_batch_end(self, n_update, logs={}):
        self.cum_update += 1  # start by index 1
        if self.eval_on_epochs:
            return
        if self.cum_update % self.each_n_epochs != 0:
            return
        if self.epoch < self.start_eval_on_epoch:
            return
        self.evaluate(self.cum_update, counter_name='iteration', logs=logs)

    def evaluate(self, epoch, counter_name='epoch', logs={}):
        # Evaluate on each set separately
        all_metrics = []

        for s in self.set_name:
            # Apply model predictions
            if self.beam_search:
                params_prediction = {'max_batch_size': self.batch_size,
                                     'n_parallel_loaders': self.extra_vars['n_parallel_loaders'],
                                     'predict_on_sets': [s],
                                     'beam_batch_size': self.beam_batch_size if
                                     self.beam_batch_size is not None else self.batch_size,
                                     'pos_unk': False,
                                     'normalize': self.normalize,
                                     'max_eval_samples': self.max_eval_samples
                                     }

                params_prediction.update(checkDefaultParamsBeamSearch(self.extra_vars))
                predictions_all = self.model_to_eval.predictBeamSearchNet(self.ds, params_prediction)[s]
            else:
                orig_size = self.extra_vars.get('eval_orig_size', False)
                params_prediction = {'batch_size': self.batch_size,
                                     'n_parallel_loaders': self.extra_vars.get('n_parallel_loaders', 8),
                                     'predict_on_sets': [s],
                                     'normalize': self.normalize,
                                     'max_eval_samples': self.max_eval_samples}
                # Convert predictions
                postprocess_fun = None
                if self.is_3DLabel:
                    postprocess_fun = [self.ds.convert_3DLabels_to_bboxes, self.extra_vars[s]['references_orig_sizes']]
                elif orig_size:
                    postprocess_fun = [self.ds.resize_semantic_output, self.extra_vars[s]['eval_orig_size_id']]
                predictions_all = \
                    self.model_to_eval.predictNet(self.ds, params_prediction, postprocess_fun=postprocess_fun)[s]

            # # Single-output model
            if not self.gt_pos or self.gt_pos == 0:
                predictions_all = [predictions_all]
                gt_positions = [0]

            # # Multi-output model
            # else:
            #     gt_positions = self.gt_pos


            # Select each output to evaluate separately
            for gt_pos, type, these_metrics, gt_id, write_type, index2word_y, index2word_x in zip(gt_positions,
                                                    self.output_types,
                                                    self.metric_name, self.gt_id, self.write_type,
                                                    self.index2word_y, self.index2word_x):

                predictions = predictions_all[gt_pos]
                trans_pred = predictions[0]
              
                # if self.verbose > 0:
                #     print('')
                #     logging.info('Prediction output ' + str(gt_pos) + ': ' + gt_id + ' ('+type+')')
                #
                # # Preprocess outputs of type text
                # if type == 'text':
                if self.gt_id[0] == 'target_text':
                    if params_prediction.get('pos_unk', False):
                        samples = trans_pred[0]
                        alphas = trans_pred[1]

                        if eval('self.ds.loaded_raw_' + s + '[0]'):
                            sources = trans_pred[2]
                        else:
                            sources = []
                            for preds in trans_pred[2]:
                                for src in preds[self.input_text_id]:
                                    sources.append(src)
                            sources = decode_predictions_beam_search(sources,
                                                                     index2word_x,
                                                                     pad_sequences=True,
                                                                     verbose=self.verbose)
                        heuristic = self.extra_vars['heuristic']
                    else:
                        samples = trans_pred
                        alphas = None
                        heuristic = None
                        sources = None
                    if self.out_pred_idx is not None:
                        samples = samples[self.out_pred_idx]
                    # Convert predictions into sentences
                    if self.beam_search:
                        trans_pred = decode_predictions_beam_search(samples,
                                                                     index2word_y,
                                                                     alphas=alphas,
                                                                     x_text=sources,
                                                                     heuristic=heuristic,
                                                                     mapping=self.extra_vars.get('mapping', None),
                                                                     verbose=self.verbose)
                    else:
                        probs = trans_pred
                        trans_pred = decode_predictions(trans_pred,
                                                         1,  # always set temperature to 1
                                                         index2word_y,
                                                         self.sampling_type,
                                                         verbose=self.verbose)

                    # Apply detokenization function if needed
                    if self.extra_vars.get('apply_detokenization', False):
                        trans_pred = map(self.extra_vars['detokenize_f'], predictions)

                    predictions[0] = trans_pred
                #
                # # Preprocess outputs of type binary
                # elif type == 'binary':
                #     predictions = decode_multilabel(predictions,
                #                                     index2word_y,
                #                                     min_val=self.min_pred_multilabel,
                #                                     verbose=self.verbose)
                #
                #     # Prepare references
                #     exec("y_raw = self.ds.Y_" + s + "[gt_id]")
                #     self.extra_vars[gt_pos][s]['references'] = self.ds.loadBinary(y_raw, gt_id)
                #
                # # Other output data types
                # else:
                #exec("self.extra_vars[gt_pos][s]['references'] = self.ds.Y_" + s)

                # Store predictions
                if self.write_samples:
                    # Store result
                    filepath = self.save_path + '/' + s + '_' + counter_name + '_' + str(epoch) + '_output_' + str(gt_pos) + '.pred'  # results file
                    if write_type == 'list':
                        list2file(filepath, predictions)
                    elif write_type == 'vqa':
                        try:
                            exec ('refs = self.ds.Y_' + s + '[gt_id]')
                        except:
                            refs = ['N/A' for _ in range(probs.shape[0])]
                        extra_data_plot = {'reference': refs,
                                           'probs': probs,
                                           'vocab': index2word_y}
                        list2vqa(filepath, predictions, self.extra_vars[gt_pos][s]['question_ids'], extra=extra_data_plot)
                    elif write_type == 'listoflists':
                        listoflists2file(filepath, predictions)
                    elif write_type == 'numpy':
                        numpy2file(filepath, predictions)
                    elif write_type == '3DLabels':
                        raise NotImplementedError('Write 3DLabels function is not implemented')
                    elif write_type == '3DSemanticLabel':
                        folder_path = self.save_path + '/' + s + '_' + counter_name + '_' + str(epoch)  # results folder
                        numpy2imgs(folder_path, predictions, eval('self.ds.X_' + s + '["' + self.input_id + '"]'), self.ds)
                    else:
                        raise NotImplementedError(
                            'The store type "' + self.write_type + '" is not implemented.')

                # Evaluate on each metric
                for metric in these_metrics:
                    if self.verbose > 0:
                        logging.info('Evaluating on metric ' + metric)
                    filepath = self.save_path + '/' + s + '.' + metric  # results file

                    if s == 'train':
                        logging.info("WARNING: evaluation results on 'train' split might be incorrect when"
                                     "applying random image shuffling.")

                    # Evaluate on the chosen metric  
                    metrics = evaluation.select[metric](
                         pred_list=predictions,
                         verbose=self.verbose,
                         extra_vars=self.extra_vars[gt_pos],
                         split=s, ds = self.ds, set=self.gt_id[0], no_ref=self.no_ref)

                    # Print results to file and store in model log
                    with open(filepath, 'a') as f:
                        header = counter_name + ','
                        line = str(epoch) + ','
                        # Store in model log
                        self.model_to_eval.log(s, counter_name, epoch)
                        for metric_ in sorted(metrics):
                            value = metrics[metric_]
                            if metric_ == 'pred':

                                filepath = self.save_path + '/' + s + '_' + counter_name + '_' + str(
                                    epoch) + '_output_' + str(gt_pos) + '.pred'
                                import numpy as np
                                np.savetxt(filepath, value, delimiter='\n', fmt='%.4f')
                            elif metric_ == 'pred_categorical':
                                
                                for threshold in value:

                                    filepath = self.save_path + '/' + s + '_' + counter_name + '_' + str(
                                        epoch) + '_threshold_'+str(float(("%0.4f"%threshold)))+ '_output_' + str(gt_pos) + '.pred'
                                    import numpy as np
                                    np.savetxt(filepath, value[threshold], delimiter='\n', fmt='%s')

                            else:
                                # Multiple-output model
                                if self.gt_pos and self.gt_pos != 0:
                                    metric_ += '_output_' + str(gt_pos)
                                all_metrics.append(metric_)
                                header += metric_ + ','
                                line += str(value) + ','
                                # Store in model log
                                self.model_to_eval.log(s, metric_, value)
                        if not self.written_header:
                            f.write(header + '\n')
                            self.written_header = True
                        f.write(line + '\n')

                    if self.verbose > 0:
                        logging.info('Done evaluating on metric ' + metric)

        # Store losses
        if logs.get('loss') is not None:
            self.model_to_eval.log('train', 'train_loss', logs['loss'])
        if logs.get('valid_loss') is not None:
            self.model_to_eval.log('val', 'val_loss', logs['valid_loss'])

        # Plot results so far
        self.do_plot = False

        if self.do_plot:
            if self.metric_name:
                self.model_to_eval.plot(counter_name, set(all_metrics), self.set_name, upperbound=self.max_plot)

        # Save the model
        if self.save_each_evaluation:
            from keras_wrapper.cnn_model import saveModel
            saveModel(self.model_to_eval, epoch, store_iter=not self.eval_on_epochs)


PrintPerformanceMetricOnEpochEndOrEachNUpdates = EvalPerformance


###################################################
# Storing callbacks
###################################################
class StoreModel(KerasCallback):
    def __init__(self, model, fun, epochs_for_save, verbose=0):
        """
        In:
            model - model to save
            fun - function for saving the model
            epochs_for_save - number of epochs before the last save
        """
        super(StoreModel, self).__init__()
        self.model_to_save = model
        self.store_function = fun
        self.epochs_for_save = epochs_for_save if epochs_for_save > 0 else np.inf
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        epoch += 1
        if epoch % self.epochs_for_save == 0:
            print('')
            self.store_function(self.model_to_save, epoch)

            # def on_batch_end(self, n_update, logs={}):
            #    n_update += 1
            #    if (n_update % self.epochs_for_save == 0):
            #        print('')
            #        self.store_function(self.model_to_save, n_update)


StoreModelWeightsOnEpochEnd = StoreModel


###################################################
# Sampling callbacks
###################################################

class Sample(KerasCallback):
    def __init__(self, model, dataset, gt_id, set_name, n_samples, each_n_updates=10000, extra_vars=None,
                 is_text=False, index2word_x=None, index2word_y=None, input_text_id=None, print_sources=False,
                 sampling='max_likelihood', temperature=1.,
                 beam_search=False, beam_batch_size=None,
                 batch_size=50, reload_epoch=0, start_sampling_on_epoch=0, is_3DLabel=False,
                 write_type='list', sampling_type='max_likelihood', out_pred_idx=None, in_pred_idx=None, verbose=1):
        """
            :param model: model to evaluate
            :param dataset: instance of the class Dataset in keras_wrapper.dataset
            :param gt_id: identifier in the Dataset instance of the output data about to evaluate
            :param metric_name: name of the performance metric
            :param set_name: name of the set split that will be evaluated
            :param n_samples: number of samples predicted during sampling
            :param each_n_updates: sampling each this number of epochs
            :param extra_vars: dictionary of extra variables
            :param is_text: defines if the predicted info is of type text
                            (in that case the data will be converted from values into a textual representation)
            :param is_3DLabel: defines if the predicted info is of type 3DLabels

            :param index2word_y: mapping from the indices to words (only needed if is_text==True)
            :param sampling: sampling mechanism used (only used if is_text==True)
            :param out_pred_idx: index of the output prediction used for evaluation
                            (only applicable if model has more than one output, else set to None)
            :param reload_epoch: number o the epoch reloaded (0 by default)
            :param start_sampling_on_epoch: only starts evaluating model if a given epoch has been reached
            :param in_pred_idx: index of the input prediction used for evaluation
                            (only applicable if model has more than one input, else set to None)
            :param verbose: verbosity level; by default 1
        """
        self.model_to_eval = model
        self.ds = dataset
        self.gt_id = gt_id
        self.index2word_x = index2word_x
        self.index2word_y = index2word_y
        self.input_text_id = input_text_id
        self.is_text = is_text
        self.sampling = sampling
        self.beam_search = beam_search
        self.beam_batch_size = beam_batch_size
        self.batch_size = batch_size
        self.set_name = set_name
        self.n_samples = n_samples
        self.each_n_updates = each_n_updates
        self.is_3DLabel = is_3DLabel
        self.extra_vars = extra_vars
        self.temperature = temperature
        self.reload_epoch = reload_epoch
        self.start_sampling_on_epoch = start_sampling_on_epoch
        self.write_type = write_type
        self.sampling_type = sampling_type
        self.out_pred_idx = out_pred_idx
        self.in_pred_idx = in_pred_idx
        self.cum_update = 0
        self.epoch_count = 0
        self.print_sources = print_sources
        self.verbose = verbose
        super(Sample, self).__init__()

    def on_epoch_end(self, n_epoch, logs={}):
        self.epoch_count += 1

    def on_batch_end(self, n_update, logs={}):
        self.cum_update += 1
        if self.epoch_count + self.reload_epoch < self.start_sampling_on_epoch:
            return
        elif self.cum_update % self.each_n_updates != 0:
            return

        # Evaluate on each set separately
        for s in self.set_name:
            if self.beam_search:
                params_prediction = {'max_batch_size': self.batch_size,
                                     'n_parallel_loaders': self.extra_vars['n_parallel_loaders'],
                                     'predict_on_sets': [s],
                                     'n_samples': self.n_samples,
                                     'pos_unk': False}
                params_prediction.update(checkDefaultParamsBeamSearch(self.extra_vars))
                predictions, truths, sources = self.model_to_eval.predictBeamSearchNet(self.ds, params_prediction)
            else:
                params_prediction = {'batch_size': self.batch_size,
                                     'n_parallel_loaders': self.extra_vars['n_parallel_loaders'],
                                     'predict_on_sets': [s],
                                     'n_samples': self.n_samples}
                # Convert predictions
                postprocess_fun = None
                if self.is_3DLabel:
                    postprocess_fun = [self.ds.convert_3DLabels_to_bboxes, self.extra_vars[s]['references_orig_sizes']]
                predictions = self.model_to_eval.predictNet(self.ds, params_prediction, postprocess_fun=postprocess_fun)

            if self.print_sources:
                if self.in_pred_idx is not None:
                    sources = [srcs[self.in_pred_idx][0] for srcs in sources]

                sources = decode_predictions_beam_search(sources,
                                                         self.index2word_x,
                                                         pad_sequences=True,
                                                         verbose=self.verbose)
            if s in predictions:
                if params_prediction['pos_unk']:
                    samples = predictions[s][0]
                    alphas = predictions[s][1]
                    heuristic = self.extra_vars['heuristic']
                else:
                    samples = predictions[s]
                    alphas = None
                    heuristic = None

                predictions = predictions[s]
                if self.is_text:
                    if self.out_pred_idx is not None:
                        samples = samples[self.out_pred_idx]
                    # Convert predictions into sentences
                    if self.beam_search:
                        predictions = decode_predictions_beam_search(samples,
                                                                     self.index2word_y,
                                                                     alphas=alphas,
                                                                     x_text=sources,
                                                                     heuristic=heuristic,
                                                                     mapping=self.extra_vars.get('mapping', None),
                                                                     verbose=self.verbose)
                    else:
                        predictions = decode_predictions(samples,
                                                         1,
                                                         self.index2word_y,
                                                         self.sampling_type,
                                                         verbose=self.verbose)
                    truths = decode_predictions_one_hot(truths, self.index2word_y, verbose=self.verbose)

                    # Apply detokenization function if needed
                    if self.extra_vars.get('apply_detokenization', False):
                        if self.print_sources:
                            sources = map(self.extra_vars['detokenize_f'], sources)
                        predictions = map(self.extra_vars['detokenize_f'], predictions)
                        truths = map(self.extra_vars['detokenize_f'], truths)

                # Write samples
                if self.print_sources:
                    # Write samples
                    for i, (source, sample, truth) in enumerate(zip(sources, predictions, truths)):
                        print("Source     (%d): %s" % (i, source))
                        print("Hypothesis (%d): %s" % (i, sample))
                        print("Reference  (%d): %s" % (i, truth))
                        print("")
                else:
                    for i, (sample, truth) in enumerate(zip(predictions, truths)):
                        print("Hypothesis (%d): %s" % (i, sample))
                        print("Reference  (%d): %s" % (i, truth))
                        print("")


SampleEachNUpdates = Sample


###################################################
# Learning modifiers callbacks
###################################################
class EarlyStopping(KerasCallback):
    """
    Applies early stopping if performance has not improved for some epochs.
    """

    def __init__(self,
                 model,
                 patience=0,
                 check_split='val',
                 metric_check='acc',
                 want_to_minimize=False,
                 eval_on_epochs=True,
                 each_n_epochs=1,
                 start_eval_on_epoch=0,
                 verbose=1):
        """
        :param model: model to check performance
        :param patience: number of beginning epochs without reduction; by default 0 (disabled)
        :param check_split: data split used to check metric value improvement
        :param metric_check: name of the metric to check
        :param verbose: verbosity level; by default 1
        """
        super(EarlyStopping, self).__init__()
        self.model_to_eval = model
        self.patience = patience
        self.check_split = check_split
        self.metric_check = metric_check
        self.eval_on_epochs = eval_on_epochs
        self.start_eval_on_epoch = start_eval_on_epoch
        self.each_n_epochs = each_n_epochs
        self.want_to_minimize = want_to_minimize

        self.verbose = verbose
        self.cum_update = 0
        self.epoch = 0

        self.threshold = self.model_to_eval.getLog(self.check_split, 'threshold')
        # check already stored scores in case we have loaded a pre-trained model
        all_scores = self.model_to_eval.getLog(self.check_split, self.metric_check)
        if self.eval_on_epochs:
            all_epochs = self.model_to_eval.getLog(self.check_split, 'epoch')
        else:
            all_epochs = self.model_to_eval.getLog(self.check_split, 'iteration')

        if all_scores[-1] is not None:
            self.best_score = max(all_scores)
            best_score_check = str(self.best_score)[:8]
            all_scores_check = [str(score)[:8] for score in all_scores]
            self.best_epoch = all_epochs[all_scores_check.index(best_score_check)]
            self.wait = max(all_epochs) - self.best_epoch
        else:
            self.best_score = -1.
            self.best_epoch = -1
            self.wait = 0

    def on_epoch_end(self, epoch, logs={}):
        epoch += 1  # start by index 1
        self.epoch = epoch
        if not self.eval_on_epochs:
            return
        elif (epoch - self.start_eval_on_epoch) % self.each_n_epochs != 0:
            return
        self.evaluate(self.epoch, counter_name='epoch')

    def on_batch_end(self, n_update, logs={}):
        self.cum_update += 1  # start by index 1
        if self.eval_on_epochs:
            return
        if self.cum_update % self.each_n_epochs != 0:
            return
        if self.epoch - self.start_eval_on_epoch < 0:
            return
        self.evaluate(self.cum_update, counter_name='update')

    def evaluate(self, epoch, counter_name='epoch'):
        current_score = self.model_to_eval.getLog(self.check_split, self.metric_check)[-1]
        threshold = self.model_to_eval.getLog(self.check_split, 'threshold')[-1]
        # Get last metric value from logs
        if current_score is None:
            warnings.warn('The chosen metric ' + str(self.metric_check) +
                          ' does not exist; the EarlyStopping callback works only with a valid metric.')
            return
        if self.want_to_minimize:
            current_score = -current_score
        # Check if the best score has been outperformed in the current epoch
        if current_score > self.best_score:
            self.best_epoch = epoch
            self.best_score = current_score
            self.threshold = threshold
            self.wait = 0
            if self.verbose > 0:
                if self.threshold is not None:
                    logging.info('---current best %s %s: %.3f threshold %.1f' % (self.check_split, self.metric_check,
                                                              current_score if not self.want_to_minimize
                                                              else -current_score, self.threshold))
                else:
                    logging.info('---current best %s %s: %.3f' % (self.check_split, self.metric_check,
                                                              current_score if not self.want_to_minimize
                                                              else -current_score))

        # Stop training if performance has not improved for self.patience epochs
        elif self.patience > 0:
            self.wait += 1
            logging.info('---bad counter: %d/%d' % (self.wait, self.patience))
            if self.wait >= self.patience:
                if self.verbose > 0:
                    if self.threshold is not None:
                        logging.info("---%s %d: early stopping. Best %s found at %s %d: %f threshold %.1f" % (
                             str(counter_name), epoch, self.metric_check, str(counter_name), self.best_epoch,
                             self.best_score if not self.want_to_minimize else -self.best_score, self.threshold))
                    else:
                        logging.info("---%s %d: early stopping. Best %s found at %s %d: %f" % ( 
                             str(counter_name), epoch, self.metric_check, str(counter_name), self.best_epoch,
                             self.best_score if not self.want_to_minimize else -self.best_score))

                self.model.stop_training = True
                exit(1)


class LearningRateReducer(KerasCallback):
    def __init__(self, reduce_rate=0.99, reduce_each_epochs=True, reduce_frequency=1, start_reduction_on_epoch=0,
                 exp_base=0.5, half_life=50000, reduction_function='linear', epsilon=1e-11, verbose=1):
        """
        Reduces learning rate during the training.
        Two different decays are implemented:
            * linear:
                lr = reduce_rate * lr
            * exponential:
                lr = exp_base^{current_step / half_life) * reduce_rate * lr

        :param reduce_rate: Reduction rate.
        :param reduce_each_epochs: Wether we reduce each epochs or each updates.
        :param reduce_frequency: Reduce each this number of epochs/updates
        :param start_reduction_on_epoch: Start reduction at this epoch
        :param exp_base: Base for exponential reduction.
        :param half_life: Half-life for exponential reduction.
        :param reduction_function: Either 'linear' or 'exponential' reduction.
        :param epsilon: Stop training if LR is below this value
        :param verbose: Be verbose.
        """

        super(LearningRateReducer, self).__init__()

        self.reduce_rate = reduce_rate
        self.reduce_each_epochs = reduce_each_epochs
        self.reduce_frequency = reduce_frequency

        self.exp_base = exp_base
        self.half_life = half_life
        self.reduction_function = reduction_function
        self.start_reduction_on_epoch = start_reduction_on_epoch
        self.verbose = verbose
        self.current_update_nb = 0
        self.epsilon = epsilon
        self.epoch = 0
        self.new_lr = None
        assert self.reduction_function in ['linear', 'exponential'], 'Reduction function "%s" unimplemented!' % \
                                                                     str(self.reduction_function)

    def on_epoch_end(self, epoch, logs={}):

        if not self.reduce_each_epochs:
            return
        elif (epoch - self.start_reduction_on_epoch) % self.reduce_frequency != 0:
            return
        self.reduce_lr(epoch)

        if float(self.new_lr) <= self.epsilon:
            if self.verbose > 0:
                logging.info('Learning rate too small, learning stops now')
            self.model.stop_training = True

    def on_batch_end(self, n_update, logs={}):

        self.current_update_nb += 1
        if self.reduce_each_epochs:
            return
        if self.current_update_nb % self.reduce_frequency != 0:
            return
        if self.epoch - self.start_reduction_on_epoch < 0:
            return
        self.reduce_lr(self.current_update_nb)

    def reduce_lr(self, current_nb):
        new_rate = self.reduce_rate if self.reduction_function == 'linear' else \
            np.power(self.exp_base, current_nb / self.half_life) * self.reduce_rate
        lr = self.model.optimizer.lr.get_value()
        self.new_lr = np.float32(lr * new_rate)
        self.model.optimizer.lr.set_value(self.new_lr)

        if self.reduce_each_epochs and self.verbose > 0:
            logging.info("LR reduction from {0:0.6f} to {1:0.6f}".format(float(lr), float(self.new_lr)))
