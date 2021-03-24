# Convert a tokenized parallel corpus into a format suitable for fast_align
# Code partially taken from:
#    https://github.com/sebastien-j/LV_groundhog/blob/master/experiments/nmt/utils/format_fast_align.py
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=str)  # Use the tokenized text files # Source
parser.add_argument("--target", type=str)  # Target
parser.add_argument("--dest", type=str)
parser.add_argument("--aligner", type=str, default='fast_align')

args = parser.parse_args()
if args.aligner == 'fast_align':
    with open(args.source, 'r') as left:
        with open(args.target, 'r') as right:
            with open(args.dest, 'w') as final:
                while True:
                    lline = left.readline()
                    rline = right.readline()
                    if (lline == '') or (rline == ''):
                        break
                    assert (lline[-1] == '\n')
                    assert (rline[-1] == '\n')
                    if (lline != '\n') and (rline != '\n'):
                        final.write(lline[:-1] + ' ||| ' + rline)
elif args.aligner == 'giza':
    raise NotImplementedError('Giza alignments still not supported')
else:
    raise AttributeError('Option %s not supported' % args.aligner)
