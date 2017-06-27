# Uses T-tables made by Chris Dyer's Fast Align
# Code adapted from: https://github.com/sebastien-j/LV_groundhog/blob/master/experiments/nmt/utils/convert_Ttables.py

import numpy as np
from keras_wrapper.extra.read_write import dict2pkl
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--fname", type=str)  # T-tables
parser.add_argument("--dest", type=str)
parser.add_argument("--verbose", type=int)

args = parser.parse_args()

d = {}
tmp_dict = dict()
with open(args.fname, 'r') as f:
    i = -1
    cur_source = -1
    for line in f:
        line = line.split()
        if line[0] != cur_source:
            i += 1
            if (i % 1000) == 0 and args.verbose > 0:
                print i
            if cur_source != -1:
                d[cur_source] = tmp_dict  # Set dict for previous word
            cur_source = line[0]
            tmp_dict = dict()
            tmp_dict[line[1]] = pow(np.e, float(line[2]))
        else:
            tmp_dict[line[1]] = pow(np.e, float(line[2]))
d[cur_source] = tmp_dict
del tmp_dict

e = {}
j = 0
for elt in d:
    if (j % 1000) == 0 and args.verbose > 0:
        print j
    j += 1
    e[elt] = sorted(d[elt], key=d[elt].get)[::-1]

f1 = {}
j = 0
for elt in e:
    if (j % 1000) == 0 and args.verbose > 0:
        print j
    j += 1
    f1[elt] = e[elt][0]

dict2pkl(f1, args.dest)
