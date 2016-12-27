"""
Extra set of callbacks.
"""

import warnings
import logging

import evaluation
from read_write import *

from keras.callbacks import Callback as KerasCallback

logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(message)s', datefmt='%d/%m/%Y %H:%M:%S')

###
# Printing callbacks
###

def checkDefaultParamsBeamSearch(params):

    required_params = ['model_inputs', 'model_outputs', 'dataset_inputs', 'dataset_outputs']
    default_params = {'beam_size': 5, 'maxlen': 30, 'normalize': False, 'alpha_factor': 1.0,
                      'words_so_far': False, 'n_parallel_loaders': 5, 'optimized_search': False,
                      'pos_unk': False}

    for k,v in params.iteritems():
        if k in default_params.keys() or k in required_params:
            default_params[k] = v

    for k in required_params:
        if k not in default_params:
            raise Exception('The beam search parameter ' + k + ' must be specified.')

    return default_params

class PrintPerformanceMetricOnEpochEndOrEachNUpdates(KerasCallback):

    def __init__(self, model, dataset, gt_id, metric_name, set_name, batch_size, each_n_epochs=1, extra_vars=dict(),
                 is_text=False, index2word_y=None, input_text_id=None, index2word_x=None, sampling='max_likelihood',
                 beam_search=False, write_samples = False, save_path='logs/pgerformance.', reload_epoch=0,
                 eval_on_epochs=True, each_n_updates=10000, start_eval_on_epoch=0,
                 write_type='list', sampling_type='max_likelihood',
                 out_pred_idx=None, early_stop=False, patience=5, stop_metric = 'Bleu-4', verbose=1):
        """
            :param model: model to evaluate
            :param dataset: instance of the class Dataset in keras_wrapper.dataset
            :param gt_id: identifier in the Dataset instance of the output data about to evaluate
            :param metric_name: name of the performance metric
            :param set_name: name of the set split that will be evaluated
            :param batch_size: batch size used during sampling
            :param each_n_epochs: sampling each this number of epochs
            :param extra_vars: dictionary of extra variables
            :param is_text: defines if the predicted info is of type text (in that case the data will be converted from values into a textual representation)
            :param index2word_y: mapping from the indices to words (only needed if is_text==True)
            :param sampling: sampling mechanism used (only used if is_text==True)
            :param write_samples: flag for indicating if we want to write the predicted data in a text file
            :param save_path: path to dumb the logs
            :param eval_on_epochs: Eval each epochs or updates
            :param start_eval_on_epoch: only starts evaluating model if a given epoch has been reached
            :param write_type: method used for writing predictions
            :param sampling_type: type of sampling used (multinomial or max_likelihood)
            :param out_pred_idx: index of the output prediction used for evaluation (only applicable if model has more than one output, else set to None)
            :param verbose: verbosity level; by default 1
        """
        self.model_to_eval = model
        self.ds = dataset
        self.gt_id = gt_id
        self.input_text_id = input_text_id
        self.index2word_x = index2word_x
        self.index2word_y = index2word_y
        self.is_text = is_text
        self.sampling = sampling
        self.beam_search = beam_search
        self.metric_name = metric_name
        self.set_name = set_name
        self.batch_size = batch_size
        self.each_n_epochs = each_n_epochs
        self.extra_vars = extra_vars
        self.save_path = save_path
        self.eval_on_epochs = eval_on_epochs
        self.start_eval_on_epoch = start_eval_on_epoch
        self.write_type = write_type
        self.sampling_type = sampling_type
        self.write_samples = write_samples
        self.out_pred_idx = out_pred_idx
        self.early_stop = early_stop
        self.patience = patience
        self.stop_metric = stop_metric
        self.best_score = -1
        self.wait = 0
        self.verbose = verbose
        self.cum_update = 0
        self.epoch = reload_epoch

    def on_epoch_end(self, epoch, logs={}):
        """
        On epoch end, sample and evaluate on the specified datasets.
        :param epoch: Current epoch number
        :param logs:
        :return:
        """
        epoch += 1 # start by index 1
        if not self.eval_on_epochs:
            return
        if epoch < self.start_eval_on_epoch:
            if self.verbose > 0:
                logging.info('Not evaluating until end of epoch '+ str(self.start_eval_on_epoch))
            return
        elif (epoch-self.start_eval_on_epoch) % self.each_n_epochs != 0:
            if self.verbose > 0:
                logging.info('Evaluating only every '+ str(self.each_n_epochs) + ' epochs')
            return
        self.evaluate(epoch, counter_name='epoch')

    def on_batch_end(self, n_update, logs={}):
        self.cum_update += 1 # start by index 1
        if self.eval_on_epochs:
            return
        if self.cum_update % self.each_n_epochs != 0:
            return
        if self.epoch < self.start_eval_on_epoch:
            return
        self.evaluate(self.cum_update, counter_name='update')

    def evaluate(self, epoch, counter_name='epoch'):
        # Evaluate on each set separately
        for s in self.set_name:
            # Apply model predictions
            params_prediction = {'batch_size': self.batch_size, 
                                 'n_parallel_loaders': self.extra_vars['n_parallel_loaders'],
                                 'predict_on_sets': [s],
                                 'pos_unk': False, 'heuristic': 0}

            if self.beam_search:
                params_prediction.update(checkDefaultParamsBeamSearch(self.extra_vars))
                predictions = self.model_to_eval.BeamSearchNet(self.ds, params_prediction)[s]
            else:
                predictions = self.model_to_eval.predictNet(self.ds, params_prediction)[s]

            if params_prediction['pos_unk']:
                samples = predictions[0]
                alphas = predictions[1]
                sources = []
                for preds in predictions[2]:
                    for src in preds[self.input_text_id]:
                        sources.append(src)
                sources = self.model_to_eval.decode_predictions_beam_search(sources,
                                                            self.index2word_x,
                                                            pad_sequences=True,
                                                            verbose=self.verbose)
                heuristic = params_prediction['heuristic']
            else:
                samples = predictions
                alphas = None
                heuristic = None
                sources = None
            if self.is_text:
                if self.out_pred_idx is not None:
                    samples = samples[self.out_pred_idx]
                # Convert predictions into sentences
                if self.beam_search:
                    predictions = self.model_to_eval.decode_predictions_beam_search(samples,
                                                                                    self.index2word_y,
                                                                                    alphas=alphas,
                                                                                    x_text=sources,
                                                                                    heuristic=heuristic,
                                                                                    verbose=self.verbose)
                else:
                    predictions = self.model_to_eval.decode_predictions(samples, 1,  # always set temperature to 1
                                                      self.index2word_y,
                                                      self.sampling_type,
                                                      verbose=self.verbose)
            # Store predictions
            if self.write_samples:
                # Store result
                filepath = self.save_path +'/'+ s +'_'+ counter_name + '_'+ str(epoch) +'.pred' # results file
                if self.write_type == 'list':
                    list2file(filepath, predictions)
                elif self.write_type == 'vqa':
                    list2vqa(filepath, predictions, self.extra_vars[s]['question_ids'])
                elif self.write_type == 'listoflists':
                    listoflists2file(filepath, predictions)
                elif self.write_type == 'numpy':
                    numpy2file(filepath, predictions)
                else:
                    raise NotImplementedError('The store type "'+self.write_type+'" is not implemented.')

            # Evaluate on each metric
            for metric in self.metric_name:
                if self.verbose > 0:
                    logging.info('Evaluating on metric '+metric)
                filepath = self.save_path +'/'+ s +'.'+metric # results file

                # Evaluate on the chosen metric
                metrics = evaluation.select[metric](
                            pred_list=predictions,
                            verbose=self.verbose,
                            extra_vars=self.extra_vars,
                            split=s)

                # Print results to file
                with open(filepath, 'a') as f:
                    header = counter_name + ','
                    line = str(epoch)+','
                    for metric_ in sorted(metrics):
                        value = metrics[metric_]
                        header += metric_ + ', '
                        line += str(value) + ', '
                    if epoch==1 or epoch==self.start_eval_on_epoch:
                        f.write(header+'\n')
                    f.write(line+'\n')
                if self.verbose > 0:
                    logging.info('Done evaluating on metric '+metric)

            # Early stop check
            if self.early_stop and s in ['val', 'validation', 'dev', 'development']:
                current_score = metrics[self.stop_metric]
                if current_score > self.best_score:
                    self.best_score = current_score
                    self.best_epoch = epoch
                    self.wait = 0
                    if self.verbose > 0:
                        logging.info('---current best %s accuracy: %.4f' % (self.stop_metric, current_score))
                else:
                    if self.wait >= self.patience:
                        if self.verbose > 0:
                            logging.info("Epoch %d: early stopping. Best %s value found at %s %d: %.4f" %
                                         (str(counter_name), epoch, self.stop_metric, self.best_epoch, self.best_score))
                            self.model.stop_training = True
                    self.wait += 1.
                    if self.verbose > 0:
                        logging.info('----bad counter: %d/%d' % (self.wait, self.patience))

class SampleEachNUpdates(KerasCallback):

    def __init__(self, model, dataset, gt_id, set_name, n_samples, each_n_updates=10000, extra_vars=dict(),
                 is_text=False, index2word_x=None, index2word_y=None, input_text_id=None, sampling='max_likelihood', beam_search=False, batch_size=50,
                 reload_epoch=0, start_sampling_on_epoch=0, write_type='list', sampling_type='max_likelihood',
                 out_pred_idx=None, in_pred_idx=None, verbose=1):
        """
        :param model: model to evaluate
        :param dataset: instance of the class Dataset in keras_wrapper.dataset
        :param gt_id: identifier in the Dataset instance of the output data about to evaluate
        :param metric_name: name of the performance metric
        :param set_name: name of the set split that will be evaluated
        :param n_samples: batch size used during sampling
        :param each_n_updates: sampling each this number of epochs
        :param extra_vars: dictionary of extra variables
        :param is_text: defines if the predicted info is of type text (in that case the data will be converted from values into a textual representation)
        :param index2word_y: mapping from the indices to words (only needed if is_text==True)
        :param sampling: sampling mechanism used (only used if is_text==True)
        :param in_pred_idx: index of the input prediction used for evaluation (only applicable if model has more than one input, else set to None)
        :param out_pred_idx: index of the output prediction used for evaluation (only applicable if model has more than one output, else set to None)
        :param reload_epoch: number o the epoch reloaded (0 by default)
        :param start_sampling_on_epoch: only starts evaluating model if a given epoch has been reached
        :param verbose: verbosity level; by default 1
        """
        self.model_to_eval = model
        self.ds = dataset
        self.gt_id = gt_id
        self.input_text_id = input_text_id
        self.index2word_x = index2word_x
        self.index2word_y = index2word_y
        self.is_text = is_text
        self.sampling = sampling
        self.beam_search = beam_search
        self.batch_size = batch_size
        self.set_name = set_name
        self.n_samples = n_samples
        self.each_n_updates = each_n_updates
        self.extra_vars = extra_vars
        self.reload_epoch = reload_epoch
        self.start_sampling_on_epoch = start_sampling_on_epoch
        self.write_type = write_type
        self.sampling_type = sampling_type
        self.out_pred_idx = out_pred_idx
        self.in_pred_idx = in_pred_idx
        self.verbose = verbose

    def on_batch_end(self, n_update, logs={}):
        n_update += 1  # start by index 1
        n_update += self.reload_epoch
        if n_update < self.start_sampling_on_epoch:
            return
        elif n_update % self.each_n_updates != 0:
            return

        # Evaluate on each set separately
        for s in self.set_name:

            # Apply model predictions
            params_prediction = {'batch_size': self.batch_size,
                                 'n_parallel_loaders': self.extra_vars['n_parallel_loaders'],
                                 'predict_on_sets': [s],
                                 'n_samples': self.n_samples,
                                 'pos_unk': False,
                                 'heuristic': 0}

            if self.beam_search:
                params_prediction.update(checkDefaultParamsBeamSearch(self.extra_vars))
                predictions, truths, sources = self.model_to_eval.BeamSearchNet(self.ds, params_prediction)[s]
            else:
                predictions, truths = self.model_to_eval.predictNet(self.ds, params_prediction)[s]
            gt_y = eval('self.ds.Y_'+s+'["'+self.gt_id+'"]')
            if params_prediction['pos_unk']:
                samples = predictions[0]
                alphas = predictions[1]
                heuristic = params_prediction['heuristic']
            else:
                samples = predictions
                alphas = None
                heuristic = None
            if self.is_text:
                if self.out_pred_idx is not None:
                    samples = samples[self.out_pred_idx]
                if self.in_pred_idx is not None:
                    sources = [srcs for srcs in sources[0][self.in_pred_idx]]

                # Convert predictions into sentences
                if self.beam_search:
                    predictions = self.model_to_eval.decode_predictions_beam_search(samples,
                                                                                    self.index2word_y,
                                                                                    alphas=alphas,
                                                                                    heuristic=heuristic,
                                                                                    verbose=self.verbose)
                    sources = self.model_to_eval.decode_predictions_beam_search(sources,
                                                                                self.index2word_x,
                                                                                pad_sequences=True,
                                                                                verbose=self.verbose)
                else:
                    predictions = self.model_to_eval.decode_predictions(samples,
                                                                        1,
                                                                        self.index2word_y,
                                                                        self.sampling_type,
                                                                        verbose=self.verbose)
                truths = self.model_to_eval.decode_predictions_one_hot(truths,
                                                                       self.index2word_y,
                                                                       verbose=self.verbose)
            # Write samples
            for i, (source, sample, truth) in enumerate(zip(sources, predictions, truths)):
                print "Source     (%d): %s"%(i, source)
                print "Hypothesis (%d): %s"%(i, sample)
                print "Reference  (%d): %s"%(i, truth)
                print ""


class ReduceLearningRate(KerasCallback):
    """
    Reduces learning rate during the training.

    Original work: jiumem [https://github.com/jiumem]
    """
    def __init__(self, patience=0, reduce_nb=10, is_early_stopping=True, verbose=1):
        """
        In:
            patience - number of beginning epochs without reduction;
                by default 0
            reduce_rate - multiplicative rate reducer; by default 0.5
            reduce_nb - maximal number of reductions performed; by default 10
            is_early_stopping - if true then early stopping is applied when
                reduce_nb is reached; by default True
            verbose - verbosity level; by default 1
        """
        super(KerasCallback, self).__init__()
        self.patience = patience
        self.wait = 0
        self.best_score = -1.
        self.current_reduce_nb = 0
        self.reduce_nb = reduce_nb
        self.is_early_stopping = is_early_stopping
        self.verbose = verbose
        self.epsilon = 0.1e-10

    def on_epoch_end(self, epoch, logs={}):
        current_score = logs.get('val_acc')
        if current_score is None:
            warnings.warn('validation score is off; ' +
                    'this reducer works only with the validation score on')
            return
        if current_score > self.best_score:
            self.best_score = current_score
            self.wait = 0
            if self.verbose > 0:
                print('---current best val accuracy: %.3f' % current_score)
        else:
            if self.wait >= self.patience:
                self.current_reduce_nb += 1
                if self.is_early_stopping:
                    if self.verbose > 0:
                        print("Epoch %d: early stopping" % epoch)
                    self.model.stop_training = True
            self.wait += 1
