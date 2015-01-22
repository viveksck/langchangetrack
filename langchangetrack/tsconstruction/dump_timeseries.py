#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""template.py: Description of what the module does."""

from argparse import ArgumentParser
import logging
import sys
import os
from os import path
from time import time
from glob import glob
import pickle
import pandas as pd
import numpy as np
import more_itertools

from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline

from joblib import Parallel, delayed

os.system("taskset -p 0xffff %d" % os.getpid())

__author__ = "Vivek Kulkarni"
__email__ = "viveksck@gmail.com"

LOGFORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"


def interpolate(x, xinter, values, finter):
    # Find all the points which we need to interpolate
    xmissing = [xm for xm in xinter if xm not in x]
    # Interpolate the function value at those points
    yintervalues = finter(xmissing)
    # Original points and values pairs
    orig_pairs = zip(x, values)
    # Interpolated points and values pairs
    interp_pairs = zip(xmissing, yintervalues)
    # Find the final values
    assert(len(orig_pairs) + len(interp_pairs) == len(xinter))
    final_pairs = sorted(orig_pairs + interp_pairs)
    return final_pairs


def create_word_time_series(old_df, new_df, w, sourcexinter, destxinter, metric_name="", interpolate=False):
    """ Create the time series for a word. """

    sourcex = np.asarray(old_df[old_df.word == w].s.values, dtype=int)
    destx = np.asarray(new_df[new_df.word == w].s.values, dtype=int)

    old_values = old_df[old_df.word == w][metric_name].values
    new_values = new_df[new_df.word == w][metric_name].values

    try:
        fold = interp1d(sourcex, old_values, bounds_error=False)
        fnew = interp1d(destx, new_values, bounds_error=False)
    except:
        print "Failed to interpolate", w
        return None, None

    if interpolate:
        final_old_pairs = interpolate(sourcex, sourcexinter, old_values, fold)
        final_new_pairs = interpolate(destx, destxinter, new_values, fnew)
        xinterold, yinterold = zip(*final_old_pairs)
        xinternew, yinternew = zip(*final_new_pairs)
    else:
        yinterold = old_values
        yinternew = new_values

    OL = [w]
    NL = [w]
    OL.extend(yinterold)
    NL.extend(yinternew)
    return (OL,  NL)


def process_chunk(chunk, func, olddf, newdf, sourcexinter, destxinter, metric_name, interpolate):
    """ Process each chunk. """
    results = [func(olddf, newdf, e, sourcexinter, destxinter, metric_name, interpolate) for e in chunk]
    return results


def main(args):
    # get the arguments
    method = args.method
    win_size = args.win_size
    step = args.step
    metric_name = args.metric_name

    # Load the data.
    L, H, olddf, newdf = pickle.load(open(args.filename))
    words = pd.Series(olddf.word.values.ravel()).unique()
    oldrows = []
    newrows = []
    sourcexrange = np.arange(args.mint, args.maxt, step)
    destxrange = np.arange(args.mint, args.maxt, step)
    if method == 'win':
        sourcexrange = sourcexrange[win_size:]
        destxrange = destxrange[:-win_size]

    if args.interpolate:
        sourcexinter = np.arange(sourcexrange[0], sourcexrange[-1] + 1, 1)
        destxinter = np.arange(destxrange[0], destxrange[-1] + 1, 1)
    else:
        sourcexinter = sourcexrange
        destxinter = destxrange

    # Construct the series
    assert(len(sourcexinter) == len(destxinter))
    words_chunks = more_itertools.chunked(words, 500)
    timeseries_chunks = Parallel(n_jobs=16, verbose=20)(delayed(process_chunk)(chunk, create_word_time_series,
                                                                               olddf, newdf,
                                                                               sourcexinter, destxinter,
                                                                               metric_name=metric_name,
                                                                               interpolate=args.interpolate) for chunk in words_chunks)

    timeseries = list(more_itertools.flatten(timeseries_chunks))

    # Dump the data frame
    for orow, newrow in timeseries:
        if orow and newrow:
            oldrows.append(orow)
            newrows.append(newrow)

    oldtimeseries = pd.DataFrame()
    newtimeseries = pd.DataFrame()
    header = ['word']
    header.extend(sourcexinter)
    newheader = ['word']
    newheader.extend(destxinter)
    oldtimeseries = oldtimeseries.from_records(oldrows, columns=header)
    oldtimeseries = oldtimeseries.fillna(method='backfill', axis=1)
    newtimeseries = newtimeseries.from_records(newrows, columns=newheader)
    newtimeseries = newtimeseries.fillna(method='backfill', axis=1)
    oldtimeseries.to_csv(args.sourcetimef, encoding='utf-8')
    newtimeseries.to_csv(args.endtimef, encoding='utf-8')


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
    parser.add_argument("-i", "--interpolate", dest="interpolate", help="interpolate", action='store_true', default=False)
    parser.add_argument("-s", "--sfile", dest="sourcetimef", help="Input file")
    parser.add_argument("-e", "--efile", dest="endtimef", help="Input file")
    parser.add_argument("-l", "--log", dest="log", help="log verbosity level", default="INFO")
    parser.add_argument("-m", "--min", dest="mint", help="starting time point", default=1900, type=int)
    parser.add_argument("-n", "--max", dest="maxt", help="ending timepoint(not included)", default=2010, type=int)
    parser.add_argument("-st", "--step", dest="step", help="stepsize", default=5, type=int)
    parser.add_argument("-me", "--method", dest="method", default="polar", help="Method to use")
    parser.add_argument("-metric", "--metric_name", dest="metric_name", default="cosine", help="Metric name to use")
    parser.add_argument("-w", "--win_size", dest="win_size", default="-1", help="Window size to use if not polar", type=int)
    args = parser.parse_args()
    if args.log == 'DEBUG':
        sys.excepthook = debug
    numeric_level = getattr(logging, args.log.upper(), None)
    logging.basicConfig(level=numeric_level, format=LOGFORMAT)
    main(args)
