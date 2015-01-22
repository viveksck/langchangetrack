#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""template.py: Description of what the module does."""

from argparse import ArgumentParser
import logging
import sys
from io import open
from os import path
from time import time
from glob import glob
import pickle

__author__ = "Vivek Kulkarni"
__email__ = "viveksck@gmail.com"

LOGFORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"

class DummyRegressor(object):
  def predict(self, X):
    return X

def main(args):
  d = DummyRegressor()
  pickle.dump(d, open('dummy_regressor.pkl', 'wb'))


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
  parser.add_argument("-l", "--log", dest="log", help="log verbosity level",
                      default="INFO")
  args = parser.parse_args()
  if args.log == 'DEBUG':
    sys.excepthook = debug
  numeric_level = getattr(logging, args.log.upper(), None)
  logging.basicConfig(level=numeric_level, format=LOGFORMAT)
  main(args)

