#!/usr/bin/env python
# -*- coding: utf-8 -*-

from argparse import ArgumentParser
import logging
import sys
from io import open
from os import path
from time import time
from glob import glob

from textblob import Blobber
from textblob_aptagger import PerceptronTagger

from collections import Counter, defaultdict
import numpy as np
import pandas as pd

__author__ = "Vivek Kulkarni"
__email__ = "viveksck@gmail.com"

LOGFORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"


def main(args):
  f = open(args.filename)
  D = {}
  tag_set = set([])
  tb = Blobber(pos_tagger=PerceptronTagger())
  for i,line in enumerate(f):
    b1 = tb(line)
    for w, t in b1.tags:
      tag_set.add(t)
      if w not in D:
        D[w] = Counter()
      D[w][t]= float(D[w][t] + 1)

  sorted_pos_tags = sorted(list(tag_set))
  rows = []
  for w in D.keys():
    row = [w]
    pos_counts_word = np.array([float(D[w][t]) for t in sorted_pos_tags])
    pos_dist_word = pos_counts_word/float(np.sum(pos_counts_word))
    assert(np.isclose(np.sum(pos_dist_word), 1.0))
    row = row + list(pos_dist_word)
    rows.append(row)

  header= ['word'] + sorted_pos_tags
  print "Set of POS tags in sorted order", header
  df = pd.DataFrame().from_records(rows,columns=header)
  print "Dumping the POS distribution."
  df.to_csv(args.outputfile, index = None, encoding='utf-8')

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
  parser.add_argument("-f", "--file", dest="filename", help="Input file")
  parser.add_argument("-o", "--outputfile", dest="outputfile", help="Output file")
  parser.add_argument("-l", "--log", dest="log", help="log verbosity level",
                      default="INFO")
  args = parser.parse_args()
  if args.log == 'DEBUG':
    sys.excepthook = debug
  numeric_level = getattr(logging, args.log.upper(), None)
  logging.basicConfig(level=numeric_level, format=LOGFORMAT)
  main(args)

