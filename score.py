# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import logging
import ast
from keras_wrapper.extra.read_write import pkl2dict
from nmt_keras import check_params
from nmt_keras.apply_model import score_corpus

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser("Use several translation models for scoring source--target pairs")
    parser.add_argument("-ds", "--dataset", required=True, help="Dataset instance with data")
    parser.add_argument("-src", "--source", required=True, help="Text file with source sentences")
    parser.add_argument("-trg", "--target", required=True, help="Text file with target sentences")
    parser.add_argument("-s", "--splits", nargs='+', required=False, default=['val'], help="Splits to sample. "
                                                                                           "Should be already included "
                                                                                           "into the dataset object.")
    parser.add_argument("-d", "--dest", required=False, help="File to save scores in")
    parser.add_argument("-v", "--verbose", required=False, action='store_true', default=False, help="Be verbose")
    parser.add_argument("-w", "--weights", nargs="*", help="Weight given to each model in the ensemble. "
                                                           "You should provide the same number of weights than models. "
                                                           "By default, it applies the same weight to each model (1/N).", default=[])
    parser.add_argument("-c", "--config", required=False, help="Config pkl for loading the model configuration. "
                                                               "If not specified, hyperparameters "
                                                               "are read from config.py")
    parser.add_argument("--models", nargs='+', required=True, help="path to the models")
    parser.add_argument("-ch", "--changes", nargs="*", help="Changes to the config. Following the syntax Key=Value",
                        default="")
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    if args.config is None:
        logger.info("Reading parameters from config.py")
        from config import load_parameters
        params = load_parameters()
    else:
        logger.info("Loading parameters from %s" % str(args.config))
        params = pkl2dict(args.config)
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
    params = check_params(params)
    score_corpus(args, params)
