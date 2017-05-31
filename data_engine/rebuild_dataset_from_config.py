import argparse
import logging
import ast
from prepare_data import build_dataset
from keras_wrapper.extra.read_write import pkl2dict

logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser("Rebuilds a dataset object from a given config instance.")
    parser.add_argument("-c", "--config", required=False, help="Config pkl for loading the model configuration. "
                                                               "If not specified, hyperparameters "
                                                               "are read from config.py")
    parser.add_argument("-ch", "--changes", nargs="*", help="Changes to the config. Following the syntax Key=Value",
                        default="")
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()
    if args.config is None:
        logging.info("Reading parameters from config.py")
        from config import load_parameters
        params = load_parameters()
    else:
        logging.info("Loading parameters from %s" % str(args.config))
        params = pkl2dict(args.config)
    try:
        for arg in args.changes:
            try:
                k, v = arg.split('=')
            except ValueError:
                print 'Overwritten arguments must have the form key=Value. \n Currently are: %s' % str(args.changes)
                exit(1)
            try:
                params[k] = ast.literal_eval(v)
            except ValueError:
                params[k] = v
    except ValueError:
        print 'Error processing arguments: (', k, ",", v, ")"
        exit(2)
    params['REBUILD_DATASET'] = True
    dataset = build_dataset(params)
