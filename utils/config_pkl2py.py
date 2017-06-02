import argparse
import ast
import sys
from keras_wrapper.extra.read_write import pkl2dict


def parse_args():
    parser = argparse.ArgumentParser("Rebuilds a python file (like config.py) from a given config instance.")
    parser.add_argument("-c", "--config", required=False, help="Config pkl for loading the model configuration. "
                                                               "If not specified, hyperparameters "
                                                               "are read from config.py")
    parser.add_argument("-d", "--dest", required=False, type=str,
                        default=None, help="Destination file. If unspecidied, standard output")
    parser.add_argument("-ch", "--changes", nargs="*", help="Changes to the config. Following the syntax Key=Value",
                        default="")
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()
    if args.config is None:
        from config import load_parameters
        params = load_parameters()
    else:
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

    if args.dest is not None:
        print args.dest
        output = open(args.dest, 'w')
    else:
        output = sys.stdout

    # Print header
    output.write('def load_parameters():\n')
    output.write('\t"""\n')
    output.write('\tLoads the defined hyperparameters\n')
    output.write('\t:return parameters: Dictionary of loaded parameters\n')
    output.write('\t"""\n')
    for key, value in params.iteritems():
        output.write('\t' + key + '=' + str(value) + '\n')
    # Print ending
    output.write('\t# ================================================ #\n')
    output.write('\tparameters = locals().copy()\n')
    output.write('\treturn parameters\n')
    if args.dest is not None:
        output.close()
