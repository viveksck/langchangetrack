#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""common_vocab.py: Dumps the common vocabulary between a set of text files."""

from argparse import ArgumentParser
import logging
import sys
from io import open
from os import path
from time import time
from glob import glob
import nltk

__author__ = "Vivek Kulkarni"
__email__ = "viveksck@gmail.com"

LOGFORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"


def main(args):
    encoding = sys.stdout.encoding or 'utf-8'
    common_vocab = None
    list_of_files = glob(args.filepattern)
    for fname in list_of_files:
        file_vocab = set()
        f = open(fname)
        for line in f:
            for sent in nltk.sent_tokenize(line):
                for word in nltk.word_tokenize(sent):
                    file_vocab.add(word)
        if common_vocab == None:
            common_vocab = file_vocab
        else:
            common_vocab = common_vocab & file_vocab
        f.close()

    for w in common_vocab:
        print w.encode(encoding)


def debug(type_, value, tb):
    if hasattr(sys, 'ps1') or not sys.stderr.isatty():
        # we are in interactive mode or we don't have a tty-like
        # device, so we call the default hook
        sys.__excepthook__(type_, value, tb)
    else:
        import traceback
        import pdb
        # we are NOT in interactive mode, print the exception...
        traceback.print_exception(type_, value, tb)
        print("\n")
        # ...then start the debugger in post-mortem mode.
        pdb.pm()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-f", "--filepattern",
                        dest="filepattern", help="Input file pattern")
    parser.add_argument("-l", "--log", dest="log", help="log verbosity level",
                        default="INFO")
    args = parser.parse_args()
    if args.log == 'DEBUG':
        sys.excepthook = debug
    numeric_level = getattr(logging, args.log.upper(), None)
    logging.basicConfig(level=numeric_level, format=LOGFORMAT)
    main(args)
