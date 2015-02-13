from argparse import ArgumentParser
import logging
import sys
import math
import operator

import numpy as np
from numpy import linalg as LA
import gensim

__author__ = "Vivek Kulkarni"
__email__ = "viveksck@gmail.com"

LOGFORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"

def main(args):
    return process(args.filename)

def process(filename):
    m = gensim.models.Word2Vec.load_word2vec_format(filename)
    print "query (ctrl-c to quit): ",
    line = sys.stdin.readline()
    while line:
        word = line.rstrip()
        print word
        tuples  = m.most_similar(word, topn=10)
        for w, s in tuples:
            print w, s
        print "----------------------------------"
        print "query (ctrl-c to quit): ",
        line = sys.stdin.readline()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--embeddings-file", dest="filename", help="embeddings file")
    parser.add_argument("-l", "--log", dest="log", help="log verbosity level",
                        default="INFO")
    args = parser.parse_args()
    if args.log == 'DEBUG':
        sys.excepthook = debug
    numeric_level = getattr(logging, args.log.upper(), None)
    logging.basicConfig(level=numeric_level, format=LOGFORMAT)
    main(args)
