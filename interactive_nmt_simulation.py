import argparse
import copy
import time
import logging
import ast
from collections import OrderedDict

from keras_wrapper.extra.read_write import pkl2dict, list2file
from utils.utils import *
from config import load_parameters
from config_online import load_parameters as load_parameters_online
from data_engine.prepare_data import build_dataset, update_dataset_from_file
from model_zoo import TranslationModel

from keras_wrapper.beam_search_interactive import InteractiveBeamSearchSampler
from keras_wrapper.cnn_model import loadModel, saveModel, updateModel
from keras_wrapper.dataset import loadDataset
from keras_wrapper.extra.isles_utils import *

logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(__name__)


def check_params(parameters):
    assert parameters['BEAM_SEARCH'], 'Only beam search is supported.'


def parse_args():
    parser = argparse.ArgumentParser("Simulate an interactive NMT session")
    parser.add_argument("-ds", "--dataset", required=True, help="Dataset instance with data")
    parser.add_argument("-s", "--splits", nargs='+', required=False, default=['val'],
                        help="Splits to sample. Should be already included into the dataset object.")
    parser.add_argument("-e", "--eval-output", required=False, help="Write evaluation results to file")
    parser.add_argument("-v", "--verbose", required=False, action='store_true', default=False, help="Be verbose")
    parser.add_argument("-c", "--config", required=False, help="Config pkl for loading the model configuration. "
                                                               "If not specified, hyperparameters "
                                                               "are read from config.py")
    parser.add_argument("--max-n", type=int, default=5, help="Maximum number of words generated between isles")
    parser.add_argument("-src", "--source", help="File of source hypothesis", required=True)
    parser.add_argument("-trg", "--references", help="Reference sentence (for simulation)", required=True)
    parser.add_argument("-d", "--dest", required=False, help="File to save translations in")
    parser.add_argument("-od", "--original-dest", help="Save original hypotheses to this file", required=False)
    parser.add_argument("-p", "--prefix", action="store_true", default=False, help="Prefix-based post-edition")
    parser.add_argument("-o", "--online",
                        action='store_true', default=False, required=False,
                        help="Online training mode after postedition. ")
    parser.add_argument("--models", nargs='+', required=True, help="path to the models")
    return parser.parse_args()


def train_online(models, dataset, source_line, target_line, params_training):
    src_seq = dataset.loadText([source_line],
                               dataset.vocabulary[params['INPUTS_IDS_DATASET'][0]],
                               params['MAX_OUTPUT_TEXT_LEN_TEST'],
                               0,
                               fill=dataset.fill_text[params['INPUTS_IDS_DATASET'][0]],
                               pad_on_batch=dataset.pad_on_batch[params['INPUTS_IDS_DATASET'][0]],
                               words_so_far=False,
                               loading_X=True)[0]
    state_below = dataset.loadText([target_line],
                                   dataset.vocabulary[params['OUTPUTS_IDS_DATASET'][0]],
                                   params['MAX_OUTPUT_TEXT_LEN_TEST'],
                                   1,
                                   fill=dataset.fill_text[params['INPUTS_IDS_DATASET'][-1]],
                                   pad_on_batch=dataset.pad_on_batch[params['INPUTS_IDS_DATASET'][-1]],
                                   words_so_far=False,
                                   loading_X=True)[0]
    trg_seq = dataset.loadTextOneHot([target_line],
                                     vocabularies=dataset.vocabulary[params['OUTPUTS_IDS_DATASET'][0]],
                                     vocabulary_len=dataset.vocabulary_len[params['OUTPUTS_IDS_DATASET'][0]],
                                     max_len=params['MAX_OUTPUT_TEXT_LEN_TEST'],
                                     offset=0,
                                     fill=dataset.fill_text[params['OUTPUTS_IDS_DATASET'][0]],
                                     pad_on_batch=dataset.pad_on_batch[params['OUTPUTS_IDS_DATASET'][0]],
                                     words_so_far=False,
                                     sample_weights=params['SAMPLE_WEIGHTS'],
                                     loading_X=False)
    for model in models:
        model.trainNetFromSamples([src_seq, state_below], trg_seq[0], params_training)


if __name__ == "__main__":

    args = parse_args()
    params = load_parameters()
    if args.config is not None:
        params = update_parameters(params, pkl2dict(args.config))

    if args.online:
        online_parameters = load_parameters_online()
        params = update_parameters(params, online_parameters)

    check_params(params)
    if args.online:
        params_training = {  # Traning params
            'n_epochs': params['MAX_EPOCH'],
            'shuffle': False,
            'batch_size': params['BATCH_SIZE'],
            'homogeneous_batches': False,
            'lr_decay': params['LR_DECAY'],
            'lr_gamma': params['LR_GAMMA'],
            'epochs_for_save': -1,
            'verbose': args.verbose,
            'eval_on_sets': params['EVAL_ON_SETS_KERAS'],
            'n_parallel_loaders': params['PARALLEL_LOADERS'],
            'extra_callbacks': [],  # callbacks,
            'reload_epoch': params['RELOAD'],
            'epoch_offset': params['RELOAD'],
            'data_augmentation': params['DATA_AUGMENTATION'],
            'patience': params.get('PATIENCE', 0),
            'metric_check': params.get('STOP_METRIC', None),
            'eval_on_epochs': params.get('EVAL_EACH_EPOCHS', True),
            'each_n_epochs': params.get('EVAL_EACH', 1),
            'start_eval_on_epoch': params.get('START_EVAL_ON_EPOCH', 0)
        }
        dataset = loadDataset(args.dataset)
        dataset = update_dataset_from_file(dataset, args.source, params,
                                           output_text_filename=args.references, splits=['train'], remove_outputs=False,
                                           compute_state_below=True)
    logger.info("<<< Using an ensemble of %d models >>>" % len(args.models))
    if args.online:
        logging.info('Loading models from %s' % str(args.models))
        model_instances = [TranslationModel(params,
                                            model_type=params['MODEL_TYPE'],
                                            verbose=params['VERBOSE'],
                                            model_name=params['MODEL_NAME'] + '_' + str(i),
                                            vocabularies=dataset.vocabulary,
                                            store_path=params['STORE_PATH'],
                                            set_optimizer=False)
                           for i in range(len(args.models))]
        models = [updateModel(model, path, -1, full_path=True) for (model, path) in zip(model_instances, args.models)]
        for nmt_model in models:
            nmt_model.setParams(params)
            nmt_model.setOptimizer()
    else:
        models = [loadModel(m, -1, full_path=True) for m in args.models]

    # Load text files
    fsrc = open(args.source, 'r')
    ftrans = open(args.dest, 'w')
    logger.info("<<< Storing corrected hypotheses into: %s >>>" % str(args.dest))

    if args.original_dest is not None:
        logger.info("<<< Storing original hypotheses into: %s >>>" % str(args.original_dest))
        ftrans_ori = open(args.original_dest, 'w')

    ftrg = open(args.references, 'r')
    target_lines = ftrg.read().split('\n')
    if target_lines[-1] == '':
        target_lines = target_lines[:-1]

    dataset = loadDataset(args.dataset)
    dataset = update_dataset_from_file(dataset, args.source, params, splits=args.splits, remove_outputs=True)

    params['INPUT_VOCABULARY_SIZE'] = dataset.vocabulary_len[params['INPUTS_IDS_DATASET'][0]]
    params['OUTPUT_VOCABULARY_SIZE'] = dataset.vocabulary_len[params['OUTPUTS_IDS_DATASET'][0]]
    # Apply sampling
    extra_vars = dict()
    extra_vars['tokenize_f'] = eval('dataset.' + params['TOKENIZATION_METHOD'])
    check_params(params)

    # Get word2index and index2word dictionaries
    index2word_y = dataset.vocabulary[params['OUTPUTS_IDS_DATASET'][0]]['idx2words']
    word2index_y = dataset.vocabulary[params['OUTPUTS_IDS_DATASET'][0]]['words2idx']
    index2word_x = dataset.vocabulary[params['INPUTS_IDS_DATASET'][0]]['idx2words']
    word2index_x = dataset.vocabulary[params['INPUTS_IDS_DATASET'][0]]['words2idx']
    unk_id = dataset.extra_words['<unk>']
    # Initialize counters
    total_errors = 0
    total_words = 0
    total_chars = 0
    total_mouse_actions = 0
    try:
        for s in args.splits:
            # Apply model predictions
            params_prediction = {'batch_size': params['BATCH_SIZE'],
                                 'n_parallel_loaders': params['PARALLEL_LOADERS'],
                                 'predict_on_sets': [s],
                                 'beam_size': params['BEAM_SIZE'],
                                 'maxlen': params['MAX_OUTPUT_TEXT_LEN_TEST'],
                                 'optimized_search': params['OPTIMIZED_SEARCH'],
                                 'model_inputs': params['INPUTS_IDS_MODEL'],
                                 'model_outputs': params['OUTPUTS_IDS_MODEL'],
                                 'dataset_inputs': params['INPUTS_IDS_DATASET'],
                                 'dataset_outputs': params['OUTPUTS_IDS_DATASET'],
                                 'normalize_probs': params['NORMALIZE_SAMPLING'],
                                 'alpha_factor': params['ALPHA_FACTOR'],
                                 'pos_unk': params['POS_UNK']}

            mapping = None if dataset.mapping == dict() else dataset.mapping

            if params['POS_UNK']:
                params_prediction['heuristic'] = params['HEURISTIC']
                input_text_id = params['INPUTS_IDS_DATASET'][0]
                vocab_src = dataset.vocabulary[input_text_id]['idx2words']
            else:
                input_text_id = None
                vocab_src = None
                mapping = None
            interactive_beam_searcher = InteractiveBeamSearchSampler(models,
                                                                     dataset,
                                                                     params_prediction,
                                                                     verbose=args.verbose)
            start_time = time.time()

            for n_line, line in enumerate(fsrc):
                errors_sentence = 0
                mouse_actions_sentence = 0
                hypothesis_number = 0
                unk_indices = []

                seqin = line.strip()
                src_seq, src_words = parse_input(seqin, dataset, word2index_x)

                logger.debug("\n \n Processing sentence %d" % (n_line + 1))
                logger.debug("Source: %s" % line[:-1])
                logger.debug("Target: %s" % target_lines[n_line])
                reference = target_lines[n_line].split()
                # 0. Get a first hypothesis
                trans_indices, costs, alphas = interactive_beam_searcher.sample_beam_search_interactive(src_seq)
                if params_prediction['pos_unk']:
                    alphas = [alphas]
                    sources = [seqin]
                    heuristic = params_prediction['heuristic']
                else:
                    alphas = None
                    heuristic = None
                    sources = None

                hypothesis = models[0].decode_predictions_beam_search([trans_indices],
                                                                      index2word_y,
                                                                      alphas=alphas,
                                                                      x_text=sources,
                                                                      heuristic=heuristic,
                                                                      mapping=mapping,
                                                                      pad_sequences=True,
                                                                      verbose=0)[0]
                # Store result
                if args.original_dest is not None:
                    filepath = args.original_dest  # results file
                    if params['SAMPLING_SAVE_MODE'] == 'list':
                        list2file(filepath, [hypothesis + '\n'], permission='a')
                    else:
                        raise Exception('Only "list" is allowed in "SAMPLING_SAVE_MODE"')
                hypothesis = hypothesis.split()
                logger.debug("Hypo_%d: %s" % (hypothesis_number, " ".join(hypothesis)))

                if hypothesis == reference:
                    # If the sentence is correct, we  validate it
                    pass
                else:
                    checked_index_r = 0
                    checked_index_h = 0
                    last_checked_index = 0
                    unk_words = []
                    fixed_words_user = OrderedDict()  # {pos: word}
                    old_isles = []
                    while checked_index_r < len(reference):
                        validated_prefix = []
                        correction_made = False
                        # Stage 1: Isles selection
                        #   1. Select the multiple isles in the hypothesis.
                        if not args.prefix:
                            hypothesis_isles = find_isles(hypothesis, reference)[0]
                            isle_indices = [(index, map(lambda x: word2index_y.get(x, unk_id), word))
                                            for index, word in hypothesis_isles]
                            logger.debug("Isles: %s" % (str(hypothesis_isles)))
                            # Count only for non selected isles
                            # Isles of length 1 account for 1 mouse action
                            mouse_actions_sentence += compute_mouse_movements(isle_indices,
                                                                              old_isles,
                                                                              last_checked_index)
                        else:
                            isle_indices = []
                        # Stage 2: Regular post editing
                        # From left to right, we will correct the hypotheses, taking into account the isles info
                        # At each timestep, the user can make two operations:
                        # Insertion of a new word at the end of the hypothesis
                        # Substitution of a word by another
                        while checked_index_r < len(reference):  # We check all words in the reference
                            new_word = reference[checked_index_r]
                            if checked_index_h >= len(hypothesis):
                                # Insertions (at the end of the sentence)
                                errors_sentence += 1
                                mouse_actions_sentence += 1
                                new_word_index = word2index_y.get(new_word, unk_id)
                                validated_prefix.append(new_word_index)
                                fixed_words_user[checked_index_h] = new_word_index
                                correction_made = True
                                if word2index_y.get(new_word) is None:
                                    unk_words.append(new_word)
                                    unk_indices.append(checked_index_h)
                                # else:
                                #    isle_indices[-1][1].append(word2index[new_word])
                                logger.debug(
                                    '"%s" to position %d (end-of-sentence)' % (str(new_word), checked_index_h))
                                last_checked_index = checked_index_h
                                break
                            elif hypothesis[checked_index_h] != reference[checked_index_r]:
                                errors_sentence += 1
                                mouse_actions_sentence += 1
                                # Substitution
                                new_word_index = word2index_y.get(new_word, unk_id)
                                fixed_words_user[checked_index_h] = new_word_index
                                validated_prefix.append(new_word_index)
                                correction_made = True
                                if word2index_y.get(new_word) is None:
                                    if checked_index_h not in unk_indices:
                                        unk_words.append(new_word)
                                        unk_indices.append(checked_index_h)
                                logger.debug('"%s" to position %d' % (str(new_word), checked_index_h))
                                last_checked_index = checked_index_h
                                break
                            else:
                                # No errors
                                new_word_index = word2index_y.get(hypothesis[checked_index_h], unk_id)
                                fixed_words_user[checked_index_h] = new_word_index
                                validated_prefix.append(new_word_index)
                                if word2index_y.get(new_word) is None:
                                    if checked_index_h not in unk_indices:
                                        unk_words.append(new_word)
                                        unk_indices.append(checked_index_h)
                                checked_index_h += 1
                                checked_index_r += 1
                                last_checked_index = checked_index_h
                        old_isles = [isle[1] for isle in isle_indices]
                        old_isles.append(validated_prefix)

                        # Generate a new hypothesis
                        if correction_made:
                            logger.debug("")
                            trans_indices, costs, alphas = interactive_beam_searcher. \
                                sample_beam_search_interactive(src_seq,
                                                               fixed_words=copy.copy(fixed_words_user),
                                                               max_N=args.max_n,
                                                               isles=isle_indices,
                                                               idx2word=index2word_y)
                            if params['POS_UNK']:
                                alphas = [alphas]
                            else:
                                alphas = None
                            hypothesis = models[0].decode_predictions_beam_search([trans_indices],
                                                                                  index2word_y,
                                                                                  alphas=alphas,
                                                                                  x_text=sources,
                                                                                  heuristic=heuristic,
                                                                                  mapping=mapping,
                                                                                  pad_sequences=True,
                                                                                  verbose=0)[0]
                            hypothesis = hypothesis.split()
                            hypothesis_number += 1
                            # UNK words management
                            if len(unk_indices) > 0:  # If we added some UNK word
                                if len(hypothesis) < len(unk_indices):  # The full hypothesis will be made up UNK words:
                                    for i, index in enumerate(range(0, len(hypothesis))):
                                        hypothesis[index] = unk_words[unk_indices[i]]
                                    for ii in range(i + 1, len(unk_words)):
                                        hypothesis.append(unk_words[ii])
                                else:  # We put each unknown word in the corresponding gap
                                    for i, index in enumerate(unk_indices):
                                        if index < len(hypothesis):
                                            hypothesis[index] = unk_words[i]
                                        else:
                                            hypothesis.append(unk_words[i])

                            logger.debug("Target: %s" % target_lines[n_line])
                            logger.debug("Hypo_%d: %s" % (hypothesis_number, " ".join(hypothesis)))
                        if hypothesis == reference:
                            break
                    # Final check: The reference is a subset of the hypothesis: Cut the hypothesis
                    if len(reference) < len(hypothesis):
                        hypothesis = hypothesis[:len(reference)]
                        errors_sentence += 1
                        logger.debug("Cutting hypothesis")

                assert hypothesis == reference, "Error: The final hypothesis does not match with the reference! \n" \
                                                "\t Split: %s \n" \
                                                "\t Sentence: %d \n" \
                                                "\t Hypothesis: %s\n" \
                                                "\t Reference: %s" % (str(s), n_line, hypothesis, reference)

                mouse_actions_sentence += 1  # This +1 is the validation action
                chars_sentence = sum(map(lambda x: len(x), hypothesis))
                total_errors += errors_sentence
                total_words += len(hypothesis)
                total_chars += chars_sentence
                total_mouse_actions += mouse_actions_sentence
                logger.debug("Final hypotesis: %s" % " ".join(hypothesis))
                logger.debug("%d errors. "
                             "Sentence WSR: %4f. "
                             "Sentence mouse strokes: %d "
                             "Sentence MAR: %4f. "
                             "Sentence MAR_c: %4f. "
                             "Accumulated (should only be considered for debugging purposes!) WSR: %4f. "
                             "MAR: %4f. "
                             "MAR_c: %4f.\n\n\n\n" %
                             (errors_sentence,
                              float(errors_sentence) / len(hypothesis),
                              mouse_actions_sentence,
                              float(mouse_actions_sentence) / len(hypothesis),
                              float(mouse_actions_sentence) / chars_sentence,
                              float(total_errors) / total_words,
                              float(total_mouse_actions) / total_words,
                              float(total_mouse_actions) / total_chars))
                if args.online:
                    train_online(models, dataset, line, " ".join(reference), params_training)

                print >> ftrans, " ".join(hypothesis)

                if (n_line + 1) % 50 == 0:
                    ftrans.flush()
                    if args.original_dest is not None:
                        ftrans_ori.flush()
                    logger.info("%d sentences processed" % (n_line + 1))
                    logger.info("Current speed is {} per sentence".format((time.time() - start_time) / (n_line + 1)))
                    logger.info("Current WSR is: %f" % (float(total_errors) / total_words))
                    logger.info("Current MAR is: %f" % (float(total_mouse_actions) / total_words))
                    logger.info("Current MAR_c is: %f" % (float(total_mouse_actions) / total_chars))

        print "Total number of errors:", total_errors
        print "Total number selections", total_mouse_actions
        print "WSR: %f" % (float(total_errors) / total_words)
        print "MAR: %f" % (float(total_mouse_actions) / total_words)
        print "MAR_c: %f" % (float(total_mouse_actions) / total_chars)

        fsrc.close()
        ftrans.close()
        if args.original_dest is not None:
            ftrans_ori.close()

    except KeyboardInterrupt:
        print 'Interrupted!'
        print "Total number of corrections (up to now):", total_errors
        print "WSR: %f" % (float(total_errors) / total_words)
        print "SR: %f" % (float(total_mouse_actions) / n_line)
