# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import ast
import logging

from config import load_parameters
from config_online import load_parameters as load_parameters_online
from data_engine.prepare_data import update_dataset_from_file
from keras_wrapper.dataset import loadDataset
from keras_wrapper.extra.callbacks import *
from nmt_keras import check_params
from nmt_keras.training import train_model, train_model_online
from utils.utils import *
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser("Train or sample NMT models")
    parser.add_argument("-c", "--config", required=False, help="Config pkl for loading the model configuration. "
                                                               "If not specified, hyperparameters "
                                                               "are read from config.py")
    parser.add_argument("-o", "--online",
                        action='store_true', default=False, required=False, help="Online training mode. ")
    parser.add_argument("-s", "--splits", nargs='+', required=False, default=['val'],
                        help="Splits to train on. Should be already included into the dataset object.")
    parser.add_argument("-ds", "--dataset", required=False, help="Dataset instance with data")
    parser.add_argument("-m", "--models", nargs='*', required=False, help="Models to load", default="")
    parser.add_argument("-src", "--source", help="File of source hypothesis", required=False)
    parser.add_argument("-trg", "--references", help="Reference sentence", required=False)
    parser.add_argument("-hyp", "--hypotheses", required=False, help="Store hypothesis to this file")
    parser.add_argument("-v", "--verbose", required=False, default=0, type=int, help="Verbosity level")
    parser.add_argument("-ch", "--changes", nargs="*", help="Changes to config, following the syntax Key=Value",
                        default="")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    parameters = load_parameters()
    if args.config is not None:
        parameters = update_parameters(parameters, pkl2dict(args.config))

    if args.online:
        online_parameters = load_parameters_online(parameters)
        parameters = update_parameters(parameters, online_parameters)
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
    if args.online:
        dataset = loadDataset(args.dataset)
        dataset = update_dataset_from_file(dataset, args.source, parameters,
                                           output_text_filename=args.references, splits=['train'], remove_outputs=False,
                                           compute_state_below=True)
        train_model_online(parameters, args.source, args.references,
                           models_path=args.models,
                           dataset=dataset,
                           stored_hypotheses_filename=args.hypotheses,
                           verbose=args.verbose)

    elif parameters['MODE'] == 'training':
        logger.info('Running training.')
        train_model(parameters, args.dataset)
        logging.info('Done!')
    elif parameters['MODE'] == 'sampling':
        logger.error('Depecrated function. For sampling from a trained model, please run sample_ensemble.py.')
        exit(2)
