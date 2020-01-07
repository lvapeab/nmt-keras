# -*- coding: utf-8 -*-
import argparse
import ast
import codecs
import logging
import os
import time
from collections import OrderedDict

from keras_wrapper.model_ensemble import InteractiveBeamSearchSampler
from keras_wrapper.extra.isles_utils import *
from keras_wrapper.online_trainer import OnlineTrainer
from nltk import ngrams
from scipy.stats import kurtosis
from sklearn.metrics.pairwise import cosine_similarity

from config import load_parameters
from config_online import load_parameters as load_parameters_online
from data_engine.prepare_data import update_dataset_from_file
from interactive_char_nmt_simulation import generate_constrained_hypothesis
from keras_wrapper.cnn_model import updateModel
from keras_wrapper.dataset import loadDataset
from keras_wrapper.extra.read_write import pkl2dict, list2file
from keras_wrapper.utils import decode_predictions_beam_search
from nmt_keras.model_zoo import TranslationModel
from nmt_keras.online_models import build_online_models
from utils.utils import update_parameters

logging.basicConfig(level=logging.DEBUG,
                    format='[%(asctime)s] %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(2)


def sampling_function(source_stream, n_samples, mode='random', src_seqs=None, hyp_seqs=None,
                      training_ngrams=None, ngram_order=4, accuracy_threshold=10,
                      cvr_centroid=None, current_n=1, cvr_model=None, sentence_mode='average', word_dim=300,
                      ibm_model=None, hypotheses=None, confidence_threshold=0.8,
                      costs=None, alphas_stream=None):
    if mode == 'random':
        indices_to_return = np.arange(len(source_stream))
        np.random.shuffle(indices_to_return)
        return indices_to_return[:n_samples], [None]
    elif mode == 'ngram-coverage':
        stream_scores = [0.] * len(source_stream)
        for i, source_sentence in enumerate(source_stream):
            inaccurate_ngram_counts = 0
            ngram_counts = 0
            for order in range(1, ngram_order + 1):
                sentence_ngrams = ngrams(source_sentence.split(), order)
                for ngram in sentence_ngrams:
                    ngram_string = u' '.join(ngram)
                    if training_ngrams.get(ngram_string) is None or training_ngrams[ngram_string] < accuracy_threshold:
                        inaccurate_ngram_counts += 1
                    ngram_counts += 1
            stream_scores[i] = float(inaccurate_ngram_counts) / ngram_counts
        indices_to_return = np.asarray(stream_scores).argsort()[::-1][:n_samples]  # Get the maximum values
        # Update ngram counts
        for i in indices_to_return:
            sentence = source_stream[i]
            for order in range(1, ngram_order + 1):
                sentence_ngrams = ngrams(sentence.split(), order)
                for ngram in sentence_ngrams:
                    ngram_string = u' '.join(ngram)
                    if training_ngrams.get(ngram_string) is not None:
                        training_ngrams[ngram_string] += 1
                    else:
                        training_ngrams[ngram_string] = 1
        return indices_to_return, [training_ngrams]

    elif mode == 'cosine-similarity':
        stream_scores = [0.] * len(source_stream)
        for i, source_sentence in enumerate(source_stream):
            sentence_representation = np.zeros(word_dim)
            for word in source_sentence.split():
                sentence_representation += cvr_model.get(word, cvr_model.get('unk', np.zeros(word_dim)))
            if sentence_mode == 'average':
                sentence_representation /= len(source_sentence.split())
            stream_scores[i] = cosine_similarity([sentence_representation], [cvr_centroid])[0, 0]
        indices_to_return = np.asarray(stream_scores).argsort()[:n_samples]  # Get the minimum values
        # Dynamically update the centroid: https://math.stackexchange.com/questions/106700/incremental-averageing
        for n, i in enumerate(indices_to_return):
            cvr_centroid += (stream_scores[i] - cvr_centroid) / (current_n + n)
        return indices_to_return, [cvr_centroid, current_n + len(indices_to_return)]
    elif mode == 'quality-estimation':
        stream_scores = [0.] * len(source_stream)
        for i, source_sentence in enumerate(source_stream):
            target_sentence = hypotheses[i]
            max_conf_probs = []
            for e in target_sentence.split():
                e_given_f_probs = ibm_model.get(e, None)
                if e_given_f_probs is not None:
                    max_conf_probs.append(max(e_given_f_probs.values()))
            num_good_aligns = 0
            for max_alignment in max_conf_probs:
                if max_alignment > confidence_threshold:
                    num_good_aligns += 1
            stream_scores[i] = 1 - float(num_good_aligns) / len(target_sentence.split())

        indices_to_return = np.asarray(stream_scores).argsort()[::-1][:n_samples]  # Get the maximum values
        # Dynamically update the centroid: https://math.stackexchange.com/questions/106700/incremental-averageing
        for n, i in enumerate(indices_to_return):
            cvr_centroid += (stream_scores[i] - cvr_centroid) / (current_n + n)
        return indices_to_return, [cvr_centroid, current_n + len(indices_to_return)]
    elif mode == 'nmt-cost':
        if hypotheses is not None:
            hypotheses_lengths = [len(hyp.split()) for hyp in hypotheses]
        else:
            hypotheses_lengths = [1.] * len(source_stream)
        stream_scores = [cost / float(length) for cost, length in zip(costs, hypotheses_lengths)]
        indices_to_return = np.asarray(stream_scores).argsort()[::-1][:n_samples]  # Get the maximum values
        return indices_to_return, [None]
    elif mode == 'attention-distraction':
        assert alphas_stream is not None, 'Cannot use sampling mode "%s" without alphas.' % str(mode)
        # Get the mean minus curtosis for each sentence
        stream_scores = [-kurtosis(alphas).mean() for alphas in alphas_stream]
        # Return the maximum values
        indices_to_return = np.asarray(stream_scores).argsort()[::-1][:n_samples]  # Get the maximum values
        return indices_to_return, [None]
    elif mode == 'coverage-penalty':
        assert alphas_stream is not None, 'Cannot use sampling mode "%s" without alphas.' % str(mode)
        # Get the coverage level for each sentence
        stream_scores = [0.] * len(source_stream)
        for i, src_seq in enumerate(src_seqs):
            alphas = np.asarray(alphas_stream[i][0])
            cp_penalty = 0.0
            for cp_j in range(len(src_seq)):
                att_weight = 0.0
                for cp_i in range(len(hyp_seqs[i])):
                    att_weight += alphas[cp_i, cp_j]
                cp_penalty += np.log(min(att_weight, 1.0))
            stream_scores[i] = float(cp_penalty) / len(src_seq)
        # Return the minimum-covered hypotheses -> Maximum penalized scores
        indices_to_return = np.asarray(stream_scores).argsort()[::-1][:n_samples]  # Get the maximum values
        return indices_to_return, [None]

    elif mode == 'query-by-committee':
        # Get the votes of the members of the committee: quality-estimation, coverage-penalty and random.
        stream_scores_qe = [0.] * len(source_stream)
        stream_scores_cp = [0.] * len(source_stream)
        assert alphas_stream is not None, 'Cannot use sampling mode "%s" without alphas.' % str(mode)
        for i, (source_sentence, src_seq) in enumerate(zip(source_stream, src_seqs)):
            # Quality estimation
            target_sentence = hypotheses[i]
            max_conf_probs = []
            for e in target_sentence.split():
                e_given_f_probs = ibm_model.get(e, None)
                if e_given_f_probs is not None:
                    max_conf_probs.append(max(e_given_f_probs.values()))
            num_good_aligns = 0
            for max_alignment in max_conf_probs:
                if max_alignment > confidence_threshold:
                    num_good_aligns += 1
            stream_scores_qe[i] = 1 - float(num_good_aligns) / len(target_sentence.split())

            # Coverage sampling
            alphas = np.asarray(alphas_stream[i][0])
            cp_penalty = 0.0
            for cp_j in range(len(src_seq)):
                att_weight = 0.0
                for cp_i in range(len(hyp_seqs[i])):
                    att_weight += alphas[cp_i, cp_j]
                cp_penalty += np.log(min(att_weight, 1.0))
            stream_scores_cp[i] = float(cp_penalty) / len(src_seq)

        # Get the mean minus curtosis for each sentence
        stream_scores_ad = [-kurtosis(alphas).mean() for alphas in alphas_stream]

        # Random sampling
        indices_to_return_random = np.arange(len(source_stream))
        np.random.shuffle(indices_to_return_random)

        indices_to_return_qe = np.asarray(stream_scores_qe).argsort()[::-1][:n_samples]  # Get the maximum values
        indices_to_return_cp = np.asarray(stream_scores_cp).argsort()[::-1][:n_samples]  # Get the maximum values
        indices_to_return_ad = np.asarray(stream_scores_ad).argsort()[::-1][:n_samples]  # Get the maximum values
        indices_to_return_random = indices_to_return_random[:n_samples]

        votes_by_sample = np.zeros(len(source_stream))

        votes_by_sample[indices_to_return_qe] += 1
        votes_by_sample[indices_to_return_cp] += 1
        votes_by_sample[indices_to_return_ad] += 1
        votes_by_sample[indices_to_return_random] += 1
        committee_votes = -votes_by_sample / 4 + np.log(votes_by_sample / 4)  # This 4 is because we have 4 members in our committee
        indices_to_return = np.asarray(committee_votes).argsort()[::-1][:n_samples]
        return indices_to_return, [None]

    else:
        raise NotImplementedError('The sampling function %s is still unimplemented' % mode)


def get_src_seq(src_line, params_prediction, tokenize_f, word2index_x):
    # Get (tokenized) input
    tokenized_input = src_line.strip()
    if params_prediction.get('apply_tokenization'):
        tokenized_input = tokenize_f(tokenized_input)
    src_seq, seq_words = parse_input(tokenized_input.encode('utf-8'), dataset, word2index_x)
    return src_seq, tokenized_input


def translate(src_seq, tokenized_input, params_prediction, index2word_y):
    # 1. Get a first hypothesis
    trans_indices, costs, alphas = interactive_beam_searcher.sample_beam_search_interactive(src_seq)
    # 1.1 Set unk replacemet strategy
    if params_prediction['pos_unk']:
        alphas = [alphas]
        sources = [tokenized_input]
        heuristic = params_prediction['heuristic']
    else:
        alphas = None
        heuristic = None
        sources = None

    # 1.2 Decode hypothesis
    hypothesis = decode_predictions_beam_search([trans_indices],
                                                index2word_y,
                                                alphas=alphas,
                                                x_text=sources,
                                                heuristic=heuristic,
                                                mapping=mapping,
                                                pad_sequences=True,
                                                verbose=0)[0]
    hypothesis = params_prediction['detokenize_f'](hypothesis) if params_prediction.get('apply_detokenization', False) else hypothesis
    return hypothesis, costs, alphas, trans_indices


def interactive_translation(src_seq, src_line, trg_line, params_prediction, args, tokenize_f,
                            index2word_y, word2index_y, index2word_x, word2index_x, unk_id,
                            total_errors, total_mouse_actions, n_line=-1, ):
    errors_sentence = 0
    mouse_actions_sentence = 0
    hypothesis_number = 0
    # Get (tokenized) input
    tokenized_reference = tokenize_f(trg_line) if args.tokenize_references else trg_line

    # Get reference as desired by the user, i.e. detokenized if necessary
    reference = params_prediction['detokenize_f'](tokenized_reference) if \
        args.detokenize_bpe else tokenized_reference

    # Detokenize line for nicer logging :)
    if args.detokenize_bpe:
        src_line = params_prediction['detokenize_f'](src_line)

    logger.debug(u'\n\nProcessing sentence %d' % n_line)
    logger.debug(u'Source: %s' % src_line)
    logger.debug(u'Target: %s' % reference)

    # 1. Get a first hypothesis
    trans_indices, costs, alphas = interactive_beam_searcher.sample_beam_search_interactive(src_seq)
    # 1.1 Set unk replacemet strategy
    if params_prediction['pos_unk']:
        alphas = [alphas]
        sources = [tokenized_input]
        heuristic = params_prediction['heuristic']
    else:
        alphas = None
        heuristic = None
        sources = None

    # 1.2 Decode hypothesis
    hypothesis = decode_predictions_beam_search([trans_indices],
                                                index2word_y,
                                                alphas=alphas,
                                                x_text=sources,
                                                heuristic=heuristic,
                                                mapping=mapping,
                                                pad_sequences=True,
                                                verbose=0)[0]
    # 1.3 Store result (optional)
    hypothesis = params_prediction['detokenize_f'](hypothesis) \
        if params_prediction.get('apply_detokenization', False) else hypothesis
    if args.original_dest is not None:
        filepath = args.original_dest  # results file
        if params_prediction['SAMPLING_SAVE_MODE'] == 'list':
            list2file(filepath, [hypothesis + '\n'], permission='a')
        else:
            raise Exception('Only "list" is allowed in "SAMPLING_SAVE_MODE"')
    logger.debug(u'Hypo_%d: %s' % (hypothesis_number, hypothesis))
    # 2.0 Interactive translation
    if hypothesis == reference:
        # 2.1 If the sentence is correct, we  validate it
        pass
    else:
        # 2.2 Wrong hypothesis -> Interactively translate the sentence
        correct_hypothesis = False
        last_correct_pos = 0
        while not correct_hypothesis:
            # 2.2.1 Empty data structures for the next sentence
            fixed_words_user = OrderedDict()
            unk_words_dict = OrderedDict()
            if not args.prefix:
                Exception(NotImplementedError, 'Segment-based interaction at'
                                               ' character level is still unimplemented')
            else:
                isle_indices = []
                unks_in_isles = []

            # 2.2.2 Compute longest common character prefix (LCCP)
            next_correction_pos, validated_prefix = common_prefix(hypothesis, reference)
            if next_correction_pos == len(reference):
                correct_hypothesis = True
                break
            # 2.2.3 Get next correction by checking against the reference
            next_correction = reference[next_correction_pos]
            # 2.2.4 Tokenize the prefix properly (possibly applying BPE)
            tokenized_validated_prefix = tokenize_f(validated_prefix + next_correction)

            # 2.2.5 Validate words
            for pos, word in enumerate(tokenized_validated_prefix.split()):
                fixed_words_user[pos] = word2index_y.get(word, unk_id)
                if word2index_y.get(word) is None:
                    unk_words_dict[pos] = word

            # 2.2.6 Constrain search for the last word
            last_user_word_pos = fixed_words_user.keys()[-1]
            if next_correction != u' ':
                last_user_word = tokenized_validated_prefix.split()[-1]
                filtered_idx2word = dict((word2index_y[candidate_word], candidate_word)
                                         for candidate_word in word2index_y if
                                         candidate_word.decode('utf-8')[:
                                         len(last_user_word)] == last_user_word)
                if filtered_idx2word != dict():
                    del fixed_words_user[last_user_word_pos]
                    if last_user_word_pos in unk_words_dict.keys():
                        del unk_words_dict[last_user_word_pos]
            else:
                filtered_idx2word = dict()

            logger.debug(u'"%s" to character %d.' % (next_correction, next_correction_pos))

            # 2.2.7 Generate a hypothesis compatible with the feedback provided by the user
            hypothesis = generate_constrained_hypothesis(interactive_beam_searcher, src_seq, fixed_words_user, params_prediction, args,
                                                         isle_indices, filtered_idx2word,
                                                         index2word_y, sources, heuristic,
                                                         mapping, unk_words_dict.keys(),
                                                         unk_words_dict.values(), unks_in_isles)
            hypothesis_number += 1
            hypothesis = u' '.join(hypothesis)  # Hypothesis is unicode
            hypothesis = params_prediction['detokenize_f'](hypothesis) \
                if args.detokenize_bpe else hypothesis
            logger.debug(u'Target: %s' % reference)
            logger.debug(u"Hypo_%d: %s" % (hypothesis_number, hypothesis))
            # 2.2.8 Add a keystroke
            errors_sentence += 1
            # 2.2.9 Add a mouse action if we moved the pointer
            if next_correction_pos - last_correct_pos > 1:
                mouse_actions_sentence += 1
            last_correct_pos = next_correction_pos

        # 2.3 Final check: The reference is a subset of the hypothesis: Cut the hypothesis
        if len(reference) < len(hypothesis):
            hypothesis = hypothesis[:len(reference)]
            errors_sentence += 1
            logger.debug("Cutting hypothesis")

    # 2.4 Security assertion
    assert hypothesis == reference, "Error: The final hypothesis does not match with the reference! \n" \
                                    "\t Split: %s \n" \
                                    "\t Sentence: %d \n" \
                                    "\t Hypothesis: %s\n" \
                                    "\t Reference: %s" % (str(s.encode('utf-8')), n_line,
                                                          hypothesis.encode('utf-8'),
                                                          reference.encode('utf-8'))

    # 3. Update user effort counters
    mouse_actions_sentence += 1  # This +1 is the validation action
    chars_sentence = len(hypothesis)
    total_errors += errors_sentence
    total_mouse_actions += mouse_actions_sentence

    # 3.1 Log some info
    logger.debug(u"Final hypotesis: %s" % hypothesis)
    logger.debug("%d errors. "
                 "Sentence WSR: %4f. "
                 "Sentence mouse strokes: %d "
                 "Sentence MAR: %4f. "
                 "Sentence MAR_c: %4f. "
                 "Sentence KSMR: %4f. "
                 %
                 (errors_sentence,
                  float(errors_sentence) / len(hypothesis),
                  mouse_actions_sentence,
                  float(mouse_actions_sentence) / len(hypothesis),
                  float(mouse_actions_sentence) / chars_sentence,
                  float(errors_sentence + mouse_actions_sentence) / chars_sentence
                  ))

    # 5 Write correct sentences into a file
    return hypothesis, total_errors, total_mouse_actions


def retrain_models(online_trainer, dataset, params, tokenized_sources, tokenized_references):
    # 4.1 Compute model inputs
    # 4.1.1 Source text -> Make a batch from the tokenized sources.
    src_seqs = dataset.loadText(tokenized_sources,
                                vocabularies=dataset.vocabulary[params['INPUTS_IDS_DATASET'][0]],
                                max_len=params['MAX_INPUT_TEXT_LEN'],
                                offset=0,
                                fill=dataset.fill_text[params['INPUTS_IDS_DATASET'][0]],
                                pad_on_batch=dataset.pad_on_batch[params['INPUTS_IDS_DATASET'][0]],
                                words_so_far=False,
                                loading_X=True)[0]

    # 4.1.2 State below
    state_below = dataset.loadText(tokenized_references,
                                   vocabularies=dataset.vocabulary[params['OUTPUTS_IDS_DATASET'][0]],
                                   max_len=params['MAX_OUTPUT_TEXT_LEN_TEST'],
                                   offset=1,
                                   fill=dataset.fill_text[params['INPUTS_IDS_DATASET'][-1]],
                                   pad_on_batch=dataset.pad_on_batch[params['INPUTS_IDS_DATASET'][-1]],
                                   words_so_far=False,
                                   loading_X=True)[0]

    # 4.1.3 Ground truth sample -> Interactively translated / post-edited sentence(s)
    trg_seqs = dataset.loadTextOneHot(tokenized_references,
                                      vocabularies=dataset.vocabulary[params['OUTPUTS_IDS_DATASET'][0]],
                                      vocabulary_len=dataset.vocabulary_len[params['OUTPUTS_IDS_DATASET'][0]],
                                      max_len=params['MAX_OUTPUT_TEXT_LEN_TEST'],
                                      offset=0,
                                      fill=dataset.fill_text[params['OUTPUTS_IDS_DATASET'][0]],
                                      pad_on_batch=dataset.pad_on_batch[params['OUTPUTS_IDS_DATASET'][0]],
                                      words_so_far=False,
                                      sample_weights=params['SAMPLE_WEIGHTS'],
                                      loading_X=False)
    # 4.2 Train online!
    online_trainer.train_online([src_seqs, state_below], trg_seqs)


def check_params(parameters):
    assert parameters['BEAM_SEARCH'], 'Only beam search is supported.'


def parse_args():
    parser = argparse.ArgumentParser("Simulate an INMT session with active learning.")
    parser.add_argument("-ds", "--dataset", required=True, help="Dataset instance with data")
    parser.add_argument("-s", "--splits", nargs='+', required=False, default=['val'],
                        help="Splits to sample. Should be already included into the dataset object.")
    parser.add_argument("-v", "--verbose", required=False, action='store_true', default=False, help="Be verbose")
    parser.add_argument("-c", "--config", required=False, help="Config pkl for loading the model configuration. "
                                                               "If not specified, hyperparameters "
                                                               "are read from config.py")
    parser.add_argument("-src", "--source", help="Stream of source sentences.", required=True)
    parser.add_argument("-trg", "--references", help="Reference sentences of the source stream (for simulation)", required=True)
    parser.add_argument("-bpe-tok", "--tokenize-bpe", help="Apply BPE tokenization", action='store_true', default=False)
    parser.add_argument("-bpe-detok", "--detokenize-bpe", help="Revert BPE tokenization",
                        action='store_true', default=True)
    parser.add_argument("-tok-ref", "--tokenize-references", help="Tokenize references. If set to False, the references"
                                                                  "should be given already preprocessed/tokenized.",
                        action='store_true', default=False)
    parser.add_argument("-d", "--dest", required=True, help="File to save translations in")
    parser.add_argument("-m", "--sampling-mode", default='random', help="Sampling mode. One of: 'random', 'ngram-coverage', 'cosine-similarity',"
                                                                        "'quality-estimation', 'nmt-cost', 'attention-distraction', 'coverage-penalty' .")
    parser.add_argument("--ngrams", help="Path to a .pkl containing ngram counts.")
    parser.add_argument("--ngram-order", type=int, default=4, help="Maximum order of ngrams used for the ngram coverage sampling mode.")
    parser.add_argument("--ngram-accuracy-threshold", type=int, default=10, help="Threshold for considering an ngram to be infrequent, for the ngram coverage sampling mode.")
    parser.add_argument("-sm", "--sentence-mode", default='average', type=str, help="Mode of computing a sentence "
                                                                                    "representation from the representations "
                                                                                    "of the words of it. "
                                                                                    "One of: 'average' and 'sum'.")
    parser.add_argument("-w", "--word-embeddings", required=False, type=str, help="Path to the processed word embeddings used for computing the representation.")
    parser.add_argument("-ctd", "--centroid", required=False, type=str, help="Path to the word embedding centroid.")
    parser.add_argument("-tau", "--confidence-threshold", required=False, type=float, help="IBM confidence estimation threshold.")
    parser.add_argument("-ibm", "--ibm-model", required=False, type=str, help="Path to the ibm model centroid.")
    parser.add_argument("-ts", "--training-samples", required=False, type=int, default=1, help="Number of training samples used to compute the word embedding centroid.")
    parser.add_argument("-p", "--prefix", action="store_true", default=True, help="Prefix-based post-edition")
    parser.add_argument("--max-n", type=int, default=5, help="Maximum number of words generated between isles")
    parser.add_argument("-pe", "--post-editing", help="Use post-editing NMT for correcting system outputs.", action='store_true', default=False)
    parser.add_argument("-bs", "--block-size", type=int, default=500, help="Number of setences to retrieve from data stream")
    parser.add_argument("-sv", "--samples-to-validate", type=int, default=50, help="Number of setences to validate at each block")
    parser.add_argument("-od", "--original-dest", help="Save original hypotheses to this file", required=False)
    parser.add_argument("-o", "--online",
                        action='store_true', default=False, required=False,
                        help="Online training mode after postedition. ")
    parser.add_argument("--models", nargs='+', required=True, help="path to the models")
    parser.add_argument("-ch", "--changes", nargs="*", help="Changes to config, following the syntax Key=Value",
                        default="")

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    # Update parameters
    if args.config is not None:
        logger.info('Reading parameters from %s.' % args.config)
        params = update_parameters({}, pkl2dict(args.config))
    else:
        logger.info('Reading parameters from config.py.')
        params = load_parameters()
    logger.info('Starting active learning with arguments: %str' % str(args))
    online_parameters = load_parameters_online(params)
    params = update_parameters(params, online_parameters)

    try:
        for arg in args.changes:
            try:
                k, v = arg.split('=')
            except ValueError:
                print ('Overwritten arguments must have the form key=Value. \n Currently are: %s' % str(args.changes))
                exit(1)
            try:
                params[k] = ast.literal_eval(v)
            except ValueError:
                params[k] = v
    except ValueError:
        print ('Error processing arguments: (', k, ",", v, ")")
        exit(2)

    check_params(params)
    logging.info("params = " + str(params))

    training_ngrams = None
    cvr_centroid = None
    cvr_samples = None
    cvr_model = None
    sentence_mode = args.sentence_mode
    word_dim = None
    ibm_model = None
    confidence_threshold = None
    return_alphas = False

    if args.sampling_mode == 'ngram-coverage':
        logging.info("Loading ngram counts from %s" % str(args.ngrams))
        training_ngrams = pkl2dict(args.ngrams)
        logging.info("Done. Counted a total of %d different ngrams " % len(training_ngrams.keys()))

    elif args.sampling_mode == 'cosine-similarity':
        logging.info("Loading word embeddings from %s" % str(args.word_embeddings))
        cvr_model = np.load(os.path.join(args.word_embeddings)).item()
        word_dim = cvr_model[cvr_model.keys()[0]].shape[0]
        logging.info("Done. Counted a total of %d word embeddings of dimension %d" % (len(cvr_model.keys()), word_dim))
        logging.info("Loading word embeddings centroid from %s" % str(args.centroid))
        cvr_centroid = np.load(args.centroid)
        cvr_samples = args.training_samples
        logging.info("Done.")

    elif args.sampling_mode == 'quality-estimation' or args.sampling_mode == 'query-by-committee':
        logging.info("Loading alingments from %s" % str(args.ibm_model))
        ibm_model = pkl2dict(args.ibm_model)
        confidence_threshold = args.confidence_threshold
        logging.info("Done.")
    elif args.sampling_mode == 'attention-distraction' or args.sampling_mode == 'coverage-penalty' or args.sampling_mode == 'query-by-committee':
        return_alphas = True

    dataset = loadDataset(args.dataset)
    dataset = update_dataset_from_file(dataset, args.source, params, splits=args.splits, remove_outputs=True)

    # Dataset backwards compatibility
    bpe_separator = dataset.BPE_separator if hasattr(dataset,
                                                     "BPE_separator") and dataset.BPE_separator is not None else '@@'
    # Set tokenization method
    params['TOKENIZATION_METHOD'] = 'tokenize_bpe' if args.tokenize_bpe else \
        params.get('TOKENIZATION_METHOD', 'tokenize_none')
    # Build BPE tokenizer if necessary
    if 'bpe' in params['TOKENIZATION_METHOD'].lower():
        logger.info('Building BPE')
        if not dataset.BPE_built:
            dataset.build_bpe(params.get('BPE_CODES_PATH', params['DATA_ROOT_PATH'] + '/training_codes.joint'),
                              separator=bpe_separator)
    # Build tokenization function
    tokenize_f = eval('dataset.' + params.get('TOKENIZATION_METHOD', 'tokenize_none'))

    # Traning params
    params_training = {  # Traning params
        'n_epochs': params['MAX_EPOCH'],
        'shuffle': False,
        'loss': params.get('LOSS', 'categorical_crossentropy'),
        'batch_size': params.get('BATCH_SIZE', 1),
        'homogeneous_batches': False,
        'optimizer': params.get('OPTIMIZER', 'SGD'),
        'lr': params.get('LR', 0.1),
        'lr_decay': params.get('LR_DECAY', None),
        'lr_gamma': params.get('LR_GAMMA', 1.),
        'epochs_for_save': -1,
        'verbose': args.verbose,
        'eval_on_sets': params['EVAL_ON_SETS_KERAS'],
        'n_parallel_loaders': params['PARALLEL_LOADERS'],
        'extra_callbacks': [],  # callbacks,
        'reload_epoch': 0,
        'epoch_offset': 0,
        'data_augmentation': params['DATA_AUGMENTATION'],
        'patience': params.get('PATIENCE', 0),
        'metric_check': params.get('STOP_METRIC', None),
        'eval_on_epochs': params.get('EVAL_EACH_EPOCHS', True),
        'each_n_epochs': params.get('EVAL_EACH', 1),
        'start_eval_on_epoch': params.get('START_EVAL_ON_EPOCH', 0),
        'additional_training_settings': {'k': params.get('K', 1),
                                         'tau': params.get('TAU', 1),
                                         'lambda': params.get('LAMBDA', 0.5),
                                         'c': params.get('C', 0.5),
                                         'd': params.get('D', 0.5)
                                         }
    }

    params['INPUT_VOCABULARY_SIZE'] = dataset.vocabulary_len[params['INPUTS_IDS_DATASET'][0]]
    params['OUTPUT_VOCABULARY_SIZE'] = dataset.vocabulary_len[params['OUTPUTS_IDS_DATASET'][0]]
    logger.info("<<< Using an ensemble of %d models >>>" % len(args.models))
    # Load trainable model(s)
    logging.info('Loading models from %s' % str(args.models))
    model_instances = [TranslationModel(params,
                                        model_type=params['MODEL_TYPE'],
                                        verbose=params['VERBOSE'],
                                        model_name=params['MODEL_NAME'] + '_' + str(i),
                                        vocabularies=dataset.vocabulary,
                                        store_path=params['STORE_PATH'],
                                        clear_dirs=False,
                                        set_optimizer=False)
                       for i in range(len(args.models))]
    models = [updateModel(model, path, -1, full_path=True) for (model, path) in zip(model_instances, args.models)]

    # Set additional inputs to models if using a custom loss function
    params['USE_CUSTOM_LOSS'] = True if 'PAS' in params['OPTIMIZER'] else False
    if params.get('N_BEST_OPTIMIZER', False):
        logging.info('Using N-best optimizer')

    models = build_online_models(models, params)
    online_trainer = OnlineTrainer(models,
                                   dataset,
                                   None,
                                   None,
                                   params_training,
                                   verbose=args.verbose)
    # Load text files
    fsrc = codecs.open(args.source, 'r', encoding='utf-8')  # File with source sentences.
    source_lines = fsrc.read().split('\n')
    if source_lines[-1] == u'':
        source_lines = source_lines[:-1]
    n_sentences = len(source_lines)
    ftrans = codecs.open(args.dest, 'w', encoding='utf-8')  # Destination file of the (post edited) translations.
    logger.info("<<< Storing corrected hypotheses into: %s >>>" % str(args.dest))

    # Do we want to save the original sentences?
    if args.original_dest is not None:
        logger.info("<<< Storing original hypotheses into: %s >>>" % str(args.original_dest))
        ftrans_ori = open(args.original_dest, 'w')
    ftrg = codecs.open(args.references, 'r', encoding='utf-8')  # File with post-edited (or reference) sentences.
    target_lines = ftrg.read().split('\n')
    if target_lines[-1] == u'':
        target_lines = target_lines[:-1]

    block_size = args.block_size

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
            params_prediction = {'max_batch_size': params['BATCH_SIZE'],
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
                                 'pos_unk': params['POS_UNK'] + return_alphas,
                                 'heuristic': params['HEURISTIC'],
                                 'search_pruning': params.get('SEARCH_PRUNING', False),
                                 'state_below_index': -1,
                                 'output_text_index': 0,
                                 'apply_tokenization': params.get('APPLY_TOKENIZATION', False),
                                 'tokenize_f': eval('dataset.' +
                                                    params.get('TOKENIZATION_METHOD', 'tokenize_none')),

                                 'apply_detokenization': params.get('APPLY_DETOKENIZATION', True),
                                 'detokenize_f': eval('dataset.' + params.get('DETOKENIZATION_METHOD',
                                                                              'detokenize_none')),
                                 'coverage_penalty': params.get('COVERAGE_PENALTY', False),
                                 'length_penalty': params.get('LENGTH_PENALTY', False),
                                 'length_norm_factor': params.get('LENGTH_NORM_FACTOR', 0.0),
                                 'coverage_norm_factor': params.get('COVERAGE_NORM_FACTOR', 0.0),
                                 'output_max_length_depending_on_x': params.get('MAXLEN_GIVEN_X', True),
                                 'output_max_length_depending_on_x_factor': params.get('MAXLEN_GIVEN_X_FACTOR', 3),
                                 'output_min_length_depending_on_x': params.get('MINLEN_GIVEN_X', True),
                                 'output_min_length_depending_on_x_factor': params.get('MINLEN_GIVEN_X_FACTOR', 2),
                                 'n_best_optimizer': params.get('N_BEST_OPTIMIZER', False)
                                 }

            # Manage pos_unk strategies
            if params['POS_UNK']:
                mapping = None if dataset.mapping == dict() else dataset.mapping
            else:
                mapping = None

            # Build interactive sampler
            interactive_beam_searcher = InteractiveBeamSearchSampler(models,
                                                                     dataset,
                                                                     params_prediction,
                                                                     excluded_words=None,
                                                                     verbose=args.verbose)
            start_time = time.time()

            if args.verbose:
                logging.info("Params prediction = " + str(params_prediction))
                if args.online:
                    logging.info("Params training = " + str(params_training))

            current_block = 0
            while n_sentences > current_block * block_size:
                logger.debug(u'\n\nProcessing block %d/%d' % (current_block, n_sentences // block_size))

                # Get sentences from stream
                source_stream = source_lines[current_block * block_size: current_block * block_size + block_size]
                target_stream = target_lines[current_block * block_size: current_block * block_size + block_size]

                src_seqs = []
                tokenized_inputs = []
                hypotheses = []
                costs = []
                alphas = []
                hyp_seqs = []

                for source_sentence in source_stream:
                    src_seq, tokenized_input = get_src_seq(source_sentence, params_prediction, tokenize_f, word2index_x)
                    src_seqs.append(src_seq)
                    tokenized_inputs.append(tokenized_input)
                    hypothesis, cost, alpha, trans_indices = translate(src_seq, tokenized_input, params_prediction, index2word_y)
                    hypotheses.append(hypothesis)
                    costs.append(cost)
                    alphas.append(alpha)
                    hyp_seqs.append(trans_indices)
                sampled_indices, updated_models = sampling_function(source_stream,
                                                                    args.samples_to_validate,
                                                                    mode=args.sampling_mode,
                                                                    src_seqs=src_seqs,
                                                                    hyp_seqs=hyp_seqs,
                                                                    training_ngrams=training_ngrams,
                                                                    ngram_order=args.ngram_order,
                                                                    accuracy_threshold=args.ngram_accuracy_threshold,
                                                                    cvr_centroid=cvr_centroid,
                                                                    current_n=cvr_samples,
                                                                    cvr_model=cvr_model,
                                                                    sentence_mode=sentence_mode,
                                                                    word_dim=word_dim,
                                                                    ibm_model=ibm_model,
                                                                    hypotheses=hypotheses,
                                                                    confidence_threshold=confidence_threshold,
                                                                    costs=costs,
                                                                    alphas_stream=alphas)

                if args.sampling_mode == 'ngram-coverage':
                    training_ngrams = updated_models[0]
                elif args.sampling_mode == 'cosine-similarity':
                    cvr_centroid = updated_models[0]
                    cvr_samples = updated_models[1]
                if not args.online:
                    x_batch = []
                    y_batch = []
                for f, source_sentence in enumerate(source_stream):
                    if f in sampled_indices:
                        if not args.post_editing:
                            hypothesis, total_errors, total_mouse_actions = interactive_translation(src_seqs[f],
                                                                                                    tokenized_inputs[f],
                                                                                                    target_stream[f],
                                                                                                    params_prediction,
                                                                                                    args,
                                                                                                    tokenize_f,
                                                                                                    index2word_y,
                                                                                                    word2index_y,
                                                                                                    index2word_x,
                                                                                                    word2index_x,
                                                                                                    unk_id,
                                                                                                    total_errors,
                                                                                                    total_mouse_actions,
                                                                                                    n_line=current_block * block_size + f)
                        else:
                            logger.debug(u'Sentence %d - Hypo_0: %s' % (current_block * block_size + f, hypotheses[f]))
                            hypothesis = params_prediction['detokenize_f'](target_stream[f]) if args.detokenize_bpe else target_stream[f]
                            logger.debug(u'Sentence %d post-edited: %s' % (current_block * block_size + f, hypothesis))

                        if args.online:
                            # Train online
                            retrain_models(online_trainer, dataset, params, [tokenized_inputs[f]], [target_stream[f]])
                        else:
                            # Add to retraining batch
                            x_batch.append(tokenized_inputs[f])
                            y_batch.append(target_stream[f])
                    else:
                        hypothesis = hypotheses[f]
                        logger.debug(u'Sentence %d automatically translated: %s' % (current_block * block_size + f, hypothesis))

                    total_words += len(hypothesis.split())
                    total_chars += len(hypothesis)
                    # 5 Write sentences into a file
                # 5 Write correct sentences into a file
                list2file(args.dest, [hypothesis], permission='a')

                if not args.online:
                    # Batched training with all sentences from Y
                    retrain_models(online_trainer, dataset, params, x_batch, y_batch)
                current_block += 1
                ftrans.flush()

            # 6. Final!
            # 6.1 Log some information
            if not args.post_editing:
                print (u"Total number of errors:", total_errors)
                print (u"Total number selections", total_mouse_actions)
                print (u"WSR: %f" % (float(total_errors) / total_words))
                print (u"MAR: %f" % (float(total_mouse_actions) / total_words))
                print (u"MAR_c: %f" % (float(total_mouse_actions) / total_chars))
                print (u"KSMR: %f" % (float(total_errors + total_mouse_actions) / total_chars))
            # 6.2 Close open files
            fsrc.close()
            ftrans.close()

    except KeyboardInterrupt:

        print (u'Interrupted!')
        if not args.post_editing:
            print (u"Total number of corrections (up to now):", total_errors)
            print (u"WSR: %f" % (float(total_errors) / total_words))
            print (u"MAR: %f" % (float(total_mouse_actions) / total_words))
            print (u"MAR_c: %f" % (float(total_mouse_actions) / total_chars))
            print (u"KSMR: %f" % (float(total_errors + total_mouse_actions) / total_chars))
