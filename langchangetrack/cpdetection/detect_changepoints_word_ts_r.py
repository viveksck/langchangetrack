from argparse import ArgumentParser

import logging
import pandas as pd
import numpy as np
import itertools
import more_itertools
import os

from functools import partial

from changepoint.utils.ts_stats import parallelize_func
from changepoint.rchangepoint import estimate_cp_pval, estimate_cp

import psutil
from multiprocessing import cpu_count

p = psutil.Process(os.getpid())
p.set_cpu_affinity(list(range(cpu_count())))

__author__ = "Vivek Kulkarni"
__email__ = "viveksck@gmail.com"

# Global variable specifying which column index the time series
# begins in a dataframe 
TS_OFFSET = 2

LOGFORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"

def normalize_timeseries(df):
    """ Centre and scale each time series column. """
    # Normalize a set of time series by subtracting the mean from each column
    # and dividing by the standard deviation.
    dfm = df.copy(deep=True)
    dfmean = df.mean()
    dfstd = df.std()
    for col in df.columns[TS_OFFSET:]:
        dfm[col] = (df[col] - dfmean[col]) / dfstd[col]
    return dfm

def get_filtered_df(df, vocab_file):
    """ Return a data frame with only the words present in the vocab file. """
    if vocab_file:
        vocab = open(vocab_file).readlines()
        vocab = [v.strip() for v in vocab]
        # Get the set of words.
        words = pd.Series(df.word.values.ravel()).unique()
        set_words = set(words)
        # Find the words common to data frame and vocab
        common_set_words = set_words & set(vocab)
        # Filter the dataframe
        df_filtered = df[df.word.isin(common_set_words)]
        return df_filtered
    else:
        return df

def get_actual_cp(df, cp_idx):
    """ 
    Return the actual time point corresponding to the change point index.
    """
    # If the cpt detection did not find any changepoint it
    # returns NAN in which case we return the same
    if np.isnan(cp_idx):
        return cp_idx

    # Add 1 as the first column is word.
    return df.columns[cp_idx + 1]

def get_pval_word_chunk(chunk, df): 
    """ 
    Process each word in a chunk and return pvalue and changepoint.
    Here we set R changepoint class = FALSE which return pvalue.

    """
    results = []
    for w in chunk:
        # Get the time series.
        ts = np.array(df[df.word == w].values[0][TS_OFFSET:])
        # Process that time series.
        results.append(estimate_cp_pval(ts))
    return results


def get_cp_word_chunk(chunk, df): 
    """ 
    Process each word in a chunk and return changepoints. Does not return 
    pvalue. 
    """
    results = []
    for w in chunk:
        ts = np.array(df[df.word == w].values[0][TS_OFFSET:])
        cp_list = estimate_cp(ts)
        if len(cp_list):
            # Returns most recent change point if any.
            results.append(cp_list[-1])
        else:
            # No change points.
            results.append(np.nan)
    return results


def main(args):
    # Read the arguments
    df_f = args.filename
    common_vocab_file = args.vocab_file
    pval_file = args.pval_file
    col_to_drop = args.col
    should_normalize = not(args.dont_normalize)
    n_jobs = int(args.workers)
    cp_pval = args.dump_pval

    print "CONFIG:"
    print "FILENAME:", df_f
    print "VOCAB FILE:", common_vocab_file
    print "PVAL_FILE:", pval_file
    print "COL TO DROP:", col_to_drop
    print "NORMALIZE:", should_normalize

    # Read the time series data
    df = pd.read_csv(df_f)
    # Restrict only to the common vocabulary.
    df = get_filtered_df(df, common_vocab_file)

    # Normalize the data frame
    if should_normalize:
        norm_df = normalize_timeseries(df)
    else:
        norm_df = df
 
    # Drop a column if needed. 
    if col_to_drop in norm_df.columns:
        cols = df.columns.tolist()
        if col_to_drop == norm_df.columns[-1]:
            time_points = cols[2:]
            new_cols = cols[0:2] + time_points[::-1]
            norm_df = norm_df[new_cols]
            print norm_df.columns  
        norm_df.drop(col_to_drop, axis = 1, inplace=True)
        
    print "Columns of the time series", norm_df.columns
    cwords = norm_df.word.values
    print "Number of words we are processing", len(cwords)

    chunksz = len(cwords)/n_jobs
    if cp_pval: 
        results = parallelize_func(cwords[:], get_pval_word_chunk, chunksz=chunksz, n_jobs=n_jobs, df = norm_df)
        cps, pvals = zip(*results)
        actual_cps = [get_actual_cp(norm_df, cp) for cp in cps]
        results = zip(cwords, actual_cps, pvals)
        header = ['word', 'cp', 'pval']
        pvalue_df = pd.DataFrame().from_records(results, columns=header)
        sdf = pvalue_df.sort(columns=['pval'])
        sdf.to_csv(pval_file, encoding='utf-8', index = None)
    else:
        results = parallelize_func(cwords[:], get_cp_word_chunk, chunksz=chunksz, n_jobs=n_jobs, df = norm_df)
        cps = results
        actual_cps = [get_actual_cp(norm_df, cp) for cp in cps]
        results = zip(cwords, actual_cps)
        header = ['word', 'cp']
        pvalue_df = pd.DataFrame().from_records(results, columns=header)
        sdf = pvalue_df.sort(columns=['cp'])
        sdf.to_csv(pval_file, encoding='utf-8', index = None)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-f", "--file", dest="filename", help="Input file")
    parser.add_argument("-v", "--vfile", dest="vocab_file", help="Input file")
    parser.add_argument("-p", "--pfile", dest="pval_file", help="Input file")
    parser.add_argument("-c", "--col", dest="col", help="Input file")
    parser.add_argument("-s", "--shuffle", dest="shuffle", action='store_true', default = False, help="Shuffle")
    parser.add_argument("-d", "--dont_normalize", dest="dont_normalize", action='store_true', default = False, help="Dont normalize")
    parser.add_argument("-w", "--workers", dest="workers", default=1, type=int, help="Number of workers to use")
    parser.add_argument("-dump_pval", "--dump_pval", dest="dump_pval",default=False, action='store_true', help="Dump pvalue or not")
    parser.add_argument("-l", "--log", dest="log", help="log verbosity level", default="INFO")
    args = parser.parse_args()
    if args.log == 'DEBUG':
        sys.excepthook = debug
    numeric_level = getattr(logging, args.log.upper(), None)
    logging.basicConfig(level=numeric_level, format=LOGFORMAT)
    main(args)
