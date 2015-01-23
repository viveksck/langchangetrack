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

def get_pval_word(df, word, use_null = True, use_balance = True, use_median = False):
    # Remove the first TS_OFFSET columns as it is 'index' and 'word'
    ts = df[df.word == word].values[0][TS_OFFSET:]
    model = MeanShiftModel() 
    stats_ts, pvals, nums = model.detect_mean_shift(ts)
    L = [word]
    L.extend(pvals)
    H = [word]
    H.extend(nums)
    return L, H

def get_pval_word_chunk(chunk, df, use_null, use_balance, use_median):
    # Get p-values for chunks of words.
    results = [get_pval_word(df, w, use_null = use_null, use_balance = use_balance, use_median = use_median) for w in chunk]
    return results

def get_min_pvalue(row):
    # Get the minimum p-value
    return np.min(row.values[1:])

def get_cp(row):
    # Add 1 to index as first column is 'word'
    idx = np.argmin(row.values[1:])
    return row.index[idx + 1]

def get_cp_pval(row, df, threshold = 0.0):
    row_series = row.values[1:]
    zscore_series = np.array(df[df.word == row.word][row.index[1:]])[0]
    assert(len(zscore_series) == len(row_series))
    sel_idx = np.where(zscore_series > threshold)[0]
    if not len(sel_idx):
        return 1.0, np.nan
    pvals_indices = np.take(row_series, sel_idx)
    min_pval = np.min(pvals_indices)
    min_idx = np.argmin(pvals_indices)
    cp_idx = sel_idx[min_idx]
    cp = row.index[1:][cp_idx]
    return min_pval, cp

def main(args):
    # Read the arguments
    df_f = args.filename
    common_vocab_file = args.vocab_file
    pval_file = args.pval_file
    sample_file = args.sample_file
    col_to_drop = args.col
    use_null = not(args.shuffle)
    use_balance = not(args.cusum)
    use_median = args.median
    should_normalize = not(args.dont_normalize)
    threshold = 1.75

    print "CONFIG:"
    print "FILENAME:", df_f
    print "VOCAB FILE:", common_vocab_file
    print "PVAL_FILE:", pval_file
    print "SAMPLE_FILE:", sample_file
    print "COL TO DROP:", col_to_drop
    print "USE NULL:", use_null
    print "USE_BALANCE:", use_balance
    print "USE_MEDIAN:", use_median
    print "NORMALIZE:", should_normalize

    df = pd.read_csv(df_f)
    df = get_filtered_df(df, common_vocab_file)

    # Normalize the data frame
    if should_normalize:
        norm_df = normalize_timeseries(df)
    else:
        norm_df = df
  
    if col_to_drop in norm_df.columns:
        cols = df.columns.tolist()
        if col_to_drop == norm_df.columns[-1]:
            time_points = cols[2:]
            new_cols = cols[0:2] + time_points[::-1]
            norm_df = norm_df[new_cols]
            print norm_df.columns  
        norm_df.drop(col_to_drop, axis = 1, inplace=True)
        

    print norm_df.columns
    cwords = norm_df.word.values
    print len(cwords)
    cwords = ['gay','sex', 'tape', 'sky', 'trees' ,'god', 'king', 'queen', 'house', 'man']
    
    results = parallelize_func(cwords[:], get_pval_word_chunk, chunksz=400, n_jobs=12, 
                               df = norm_df, use_null = use_null, use_balance = use_balance, 
                               use_median = use_median)

    pvals, num_samples = zip(*results)

    header = ['word'] + list(norm_df.columns[TS_OFFSET:len(pvals[0]) + 1])
    pvalue_df = pd.DataFrame().from_records(list(pvals), columns=header)
    
    # Append additonal columns to the final df
    pvalue_df_final = pvalue_df.copy(deep = True)

    pvalue_df_final['min_pval'] = pvalue_df.apply(get_min_pvalue, axis=1)
    pvalue_df_final['cp'] = pvalue_df.apply(get_cp, axis=1)
    pvalue_df_final['tpval'], pvalue_df_final['tcp'] = zip(*pvalue_df.apply(get_cp_pval, axis = 1, df = norm_df, threshold = threshold))

    # Write the output.
    num_samples_df = pd.DataFrame().from_records(list(num_samples), columns = header)
    num_samples_df.to_csv(sample_file, encoding='utf-8')

    sdf = pvalue_df_final.sort(columns=['tpval'])
    sdf.to_csv(pval_file, encoding='utf-8')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-f", "--file", dest="filename", help="Input file")
    parser.add_argument("-v", "--vfile", dest="vocab_file", help="Input file")
    parser.add_argument("-p", "--pfile", dest="pval_file", help="Input file")
    parser.add_argument("-n", "--nfile", dest="sample_file", help="Input file")
    parser.add_argument("-c", "--col", dest="col", help="Input file")
    parser.add_argument("-s", "--shuffle", dest="shuffle", action='store_true', default = False, help="Shuffle")
    parser.add_argument("-d", "--dont_normalize", dest="dont_normalize", action='store_true', default = False, help="Dont normalize")
    parser.add_argument("-csum", "--cusum", dest="cusum", action='store_true', default = False, help="use cusum")
    parser.add_argument("-median", "--median", dest="median", action='store_true', default = False, help="use median")
    parser.add_argument("-test", "--test", dest="test", action='store_true', default = False, help="just test")
    parser.add_argument("-l", "--log", dest="log", help="log verbosity level", default="INFO")
    args = parser.parse_args()
    if args.log == 'DEBUG':
        sys.excepthook = debug
    numeric_level = getattr(logging, args.log.upper(), None)
    logging.basicConfig(level=numeric_level, format=LOGFORMAT)
    main(args)
