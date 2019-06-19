import argparse
import logging
import sys
import os
from keras_wrapper.utils import average_models
sys.path.insert(1, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("../"))

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser("Averages models")

    parser.add_argument("-d", "--dest",
                        default='./model',
                        required=False,
                        help="Path to the averaged model. If not specified, the model is saved in './model'.")
    parser.add_argument("-v", "--verbose", required=False, default=0, type=int, help="Verbosity level")
    parser.add_argument("-w", "--weights", nargs="*", help="Weight given to each model in the averaging. You should provide the same number of weights than models."
                                                           "By default, it applies the same weight to each model (1/N).", default=[])
    parser.add_argument("-m", "--models", nargs="+", required=True, help="Path to the models")
    return parser.parse_args()


def weighted_average(args):
    """
    Apply a weighted average to the models.
    :param args: Options for the averaging function:
              * models: Path to the models.
              * dest: Path to the averaged model. If unspecified, the model is saved in './model'
              * weights: Weight given to each model in the averaging. Should be the same number of weights than models.
                         If unspecified, it applies the same weight to each model (1/N).
    :return:
    """
    logger.info("Averaging %d models" % len(args.models))
    average_models(args.models, args.dest, weights=args.weights)
    logger.info('Averaging finished.')


if __name__ == "__main__":

    args = parse_args()
    weighted_average(args)
