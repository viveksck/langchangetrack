from argparse import ArgumentParser
import logging
import pandas as pd
import numpy as np
import itertools
import more_itertools
import os

from functools import partial
from changepoint.mean_shift_model import MeanShiftModel
from changepoint.utils.ts_stats import parallelize_func

os.system("taskset -p 0xffff %d" % os.getpid())

__author__ = "Vivek Kulkarni"
__email__ = "viveksck@gmail.com"

# Global variable specifying which column index the time series
# begins in a dataframe
TS_OFFSET = 2

LOGFORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"


def normalize_timeseries(df):
    """ 
        Normalize each column of the data frame by its mean and standard
        deviation. 
    """
    dfm = df.copy(deep=True)
    dfmean = df.mean()
    dfstd = df.std()
    for col in df.columns[2:]:
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


def get_pval_word(df, word):
    """ 
    Get the pvalue of a change point at each time point 't' corresponding to
    the word. Also return the number of tail successes during boot strap.
    Use a mean shift model for this.
    """
    # Remove the first TS_OFFSET columns as it is 'index' and 'word' to get the
    # time series for that word.
    ts = df[df.word == word].values[0][TS_OFFSET:]
    # Create a mean shift model
    model = MeanShiftModel()
    # Detect the change points using a mean shift model
    stats_ts, pvals, nums = model.detect_mean_shift(ts)
    # Return the word and pvals associated with each time point.
    L = [word]
    L.extend(pvals)
    H = [word]
    H.extend(nums)
    return L, H


def get_pval_word_chunk(chunk, df):
    """ Get the p-values for each time point for a chunk of words. """
    results = [get_pval_word(df, w) for w in chunk]
    return results


def get_minpval_cp(pvalue_df_row):
    """ 
    Get the minimum p-value and the corresponding time point for each word.
    """
    # first column is 'word', so ignore it
    index_series = pvalue_df_row.index[1:]
    row_series = pvalue_df_row.values[1:]
    assert(len(index_series) == len(row_series))

    # Find the minimum pvalue
    min_pval = np.min(row_series)
    # Find the index where the minimum pvalue occurrs.
    min_idx = np.argmin(row_series)
    # Get the timepoint corresponding to that index
    min_cp = index_series[min_idx]

    return min_pval, min_cp


def get_cp_pval(pvalue_df_row, zscore_df, threshold=0.0):
    """
        Get the minimum p-value corresponding timepoint which also has 
        a Z-SCORE > threshold.

    """
    # First column is 'word', so ignore it
    row_series = pvalue_df_row.values[1:]
    # Corresponding Z-Score series for the exact same set of timepoints.
    zscore_series = np.array(zscore_df[zscore_df.word == pvalue_df_row.word][pvalue_df_row.index[1:]])[0]
    assert(len(zscore_series) == len(row_series))

    # Get all the indices where zscore exceeds a threshold
    sel_idx = np.where(zscore_series > threshold)[0]
    # If there are no such indices return NAN
    if not len(sel_idx):
        return 1.0, np.nan

    # We have some indices. Select the pvalues for those indices.
    pvals_indices = np.take(row_series, sel_idx)
    # Find the minimum pvalue among those candidates.
    min_pval = np.min(pvals_indices)
    # Find the minimum candidate index corresponding to that pvalue
    min_idx = np.argmin(pvals_indices)
    # Select the actual index that it corresponds to
    cp_idx = sel_idx[min_idx]
    # Translate that to the actual timepoint and return it.
    cp = pvalue_df_row.index[1:][cp_idx]
    return min_pval, cp


def main(args):
    # Read the arguments
    df_f = args.filename
    common_vocab_file = args.vocab_file
    pval_file = args.pval_file
    sample_file = args.sample_file
    col_to_drop = args.col
    should_normalize = not(args.dont_normalize)
    threshold = float(args.threshold)

    print "Config:"
    print "Input data frame file name:", df_f
    print "Vocab file", common_vocab_file
    print "Output pvalue file", pval_file
    print "Output sample file", sample_file
    print "Columns to drop", col_to_drop
    print "Normalize Time series:", should_normalize
    print "Threshold", threshold

    # Read the time series data
    df = pd.read_csv(df_f)
    # Consider only words in the common vocabulary.
    df = get_filtered_df(df, common_vocab_file)

    # Normalize the data frame
    if should_normalize:
        norm_df = normalize_timeseries(df)
    else:
        norm_df = df

    # Drop the column if needed. We typically drop the 1st column as it always is 0 by
    # default.
    if col_to_drop in norm_df.columns:
        cols = df.columns.tolist()
        if col_to_drop == norm_df.columns[-1]:
            time_points = cols[2:]
            new_cols = cols[0:2] + time_points[::-1]
            norm_df = norm_df[new_cols]
        norm_df.drop(col_to_drop, axis=1, inplace=True)
        print "Dropped column", col_to_drop

    print "Columns of the data frame are", norm_df.columns
    cwords = norm_df.word.values
    print "Number of words we are analyzing:", len(cwords)

    results = parallelize_func(cwords[:], get_pval_word_chunk, chunksz=400, n_jobs=12, df=norm_df)

    pvals, num_samples = zip(*results)

    header = ['word'] + list(norm_df.columns[TS_OFFSET:len(pvals[0]) + 1])
    pvalue_df = pd.DataFrame().from_records(list(pvals), columns=header)

    # Append additonal columns to the final df
    pvalue_df_final = pvalue_df.copy(deep=True)

    pvalue_df_final['min_pval'], pvalue_df_final['cp'] = zip(*pvalue_df.apply(get_minpval_cp, axis=1))
    pvalue_df_final['tpval'], pvalue_df_final['tcp'] = zip(*pvalue_df.apply(get_cp_pval, axis=1, zscore_df=norm_df, threshold=threshold))

    # Write the pvalue output.
    num_samples_df = pd.DataFrame().from_records(list(num_samples), columns=header)
    num_samples_df.to_csv(sample_file, encoding='utf-8')

    # Write the sample output
    sdf = pvalue_df_final.sort(columns=['tpval'])
    sdf.to_csv(pval_file, encoding='utf-8')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-f", "--file", dest="filename", help="Input time series file")
    parser.add_argument("-v", "--vfile", dest="vocab_file", help="Common Vocab file")
    parser.add_argument("-p", "--pfile", dest="pval_file", help="Output pvalue file")
    parser.add_argument("-n", "--nfile", dest="sample_file", help="Output sample file")
    parser.add_argument("-c", "--col", dest="col", help="column to drop")
    parser.add_argument("-d", "--dont_normalize", dest="dont_normalize", action='store_true', default=False, help="Dont normalize")
    parser.add_argument("-t", "--threshold", dest="threshold", default=1.75, type=float, help="column to drop")
    parser.add_argument("-l", "--log", dest="log", help="log verbosity level", default="INFO")
    args = parser.parse_args()
    if args.log == 'DEBUG':
        sys.excepthook = debug
    numeric_level = getattr(logging, args.log.upper(), None)
    logging.basicConfig(level=numeric_level, format=LOGFORMAT)
    main(args)
