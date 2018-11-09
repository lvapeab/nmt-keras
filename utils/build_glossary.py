# -*- coding: utf-8 -*-
from __future__ import print_function
from keras_wrapper.extra.read_write import dict2pkl
import argparse

# Preprocess a glossary file with the format
#   word <separator> desired_replacement
# and stores them in a suitable format (.pkl)


def build_glossary(glossary_text_file, dest_filename, separator='\t'):
    """
    Preprocess a glossary file with the format
        word <separator> desired_replacement
    and stores them in a suitable format (.pkl)

    :param glossary_text_file: Path to the glossary file.
    :param dest_filename: Output filename.
    :param separator: Separator between words and replacements
    """
    glossary = dict()
    print ("Reading glossary from %s" % glossary_text_file)
    for glossary_line in open(glossary_text_file).read().splitlines():
        split_line = glossary_line.split(separator)
        glossary[split_line[0]] = ' '.join(split_line[1:])
    print ("Done. Saving glossary into %s" % dest_filename)
    dict2pkl(glossary, dest_filename)


def parse_args():
    parser = argparse.ArgumentParser("Process a glossary text file. A glossary file is a mapping of words specifying "
                                     "its translation.")
    parser.add_argument("-g", "--glossary", required=True, help="Glossary text file. A word mapping per line.")
    parser.add_argument("-d", "--destination", required=True, help="Destination file.", default='word2vec.en')
    parser.add_argument("-s", "--separator", required=False, default=' ', help="Separator of the glossary file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_glossary(args.glossary, args.destination, separator=args.separator)
