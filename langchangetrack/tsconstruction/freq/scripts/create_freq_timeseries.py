#!/usr/bin/env python
# -*- coding: utf-8 -*-

from argparse import ArgumentParser
import logging
import sys
from io import open
from os import path
from time import time
from glob import glob
import numpy as np
import pandas as pd

__author__ = "Vivek Kulkarni"
__email__ = "viveksck@gmail.com"

LOGFORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"


def main(args):
    # Read the input arguments.
    inputdir = args.inputdir
    start = args.start
    end = args.end
    step = args.step
    timepoints = np.arange(start, end, step)
    timepoints = [str(timepoint) for timepoint in timepoints]
    num = int(args.num)
    freq = args.freq

    # Normalize the frequencies.
    normdf = None
    dfs = (pd.read_table(path.join(inputdir, timepoint + '.freq'), sep=' ',
quotechar=' ', names=['word', timepoint]) for timepoint in (timepoints))
    for i, df in enumerate(dfs):
        df[str(timepoints[i])] = df[str(timepoints[i])] / df[str(timepoints[i])].sum()
        if normdf is None:
            normdf = df[:num]
            continue
        df = df[:num]
        normdf = pd.merge(normdf, df, on='word', how='outer')

    # Convert them to log scale becoz that is what matters !
    if args.log10:
        for timepoint in timepoints:
            normdf[timepoint] = np.log10(normdf[timepoint])

    normdf.to_csv(freq, encoding='utf-8')


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
    parser.add_argument("-d", "--inputdir", dest="inputdir", help="Input file")
    parser.add_argument("-s", "--start", dest="start", help="start time", type=int)
    parser.add_argument("-e", "--end", dest="end", help="end time(not included)", type=int)
    parser.add_argument("-p", "--step", dest="step", help="step", type=int)
    parser.add_argument("-num", "--num", dest="num", help="Number of words topN", type=int, default=30000)
    parser.add_argument("-f", "--freq", dest="freq", help="Output freq dist file")
    parser.add_argument("-log", "--log10", dest="log10", action="store_true", default=False,  help="freq")
    parser.add_argument("-l", "--log", dest="log", help="log verbosity level",
                        default="INFO")
    args = parser.parse_args()
    if args.log == 'DEBUG':
        sys.excepthook = debug
    numeric_level = getattr(logging, args.log.upper(), None)
    logging.basicConfig(level=numeric_level, format=LOGFORMAT)
    main(args)
