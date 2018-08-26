# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import ast
import logging

from keras_wrapper.extra.read_write import pkl2dict
from config import load_parameters
from utils.utils import update_parameters
from nmt_keras import check_params
from nmt_keras.training import train_model


def parse_args():
    parser = argparse.ArgumentParser("Train or sample NMT models")
    parser.add_argument("-c", "--config", required=False, help="Config pkl for loading the model configuration. "
                                                               "If not specified, hyperparameters "
                                                               "are read from config.py")
    parser.add_argument("-ds", "--dataset", required=False, help="Dataset instance with data")
    parser.add_argument("changes", nargs="*", help="Changes to config. "
                                                   "Following the syntax Key=Value",
                        default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    parameters = load_parameters()
    if args.config is not None:
        parameters = update_parameters(parameters, pkl2dict(args.config))
    try:
        for arg in args.changes:
            try:
                k, v = arg.split('=')
            except ValueError:
                print ('Overwritten arguments must have the form key=Value. \n Currently are: %s' % str(args.changes))
                exit(1)
            try:
                parameters[k] = ast.literal_eval(v)
            except ValueError:
                parameters[k] = v
    except ValueError:
        print ('Error processing arguments: (', k, ",", v, ")")
        exit(2)

    parameters = check_params(parameters)
    if parameters['MODE'] == 'training':
        logging.info('Running training.')
        train_model(parameters, args.dataset)
    elif parameters['MODE'] == 'sampling':
        logging.error('Depecrated function. For sampling from a trained model, please run sample_ensemble.py.')
        exit(2)
    logging.info('Done!')
