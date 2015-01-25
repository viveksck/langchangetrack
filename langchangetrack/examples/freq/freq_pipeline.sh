INPUT_DIR=$1
WORKING_DIR=$2
OUTPUT_DIR=$3
STARTTIMEPOINT=$4
ENDTIMEPOINT=$5
STEP=$6
FILTER_VOCAB_FILE=$7
EXT=ngrams
mkdir -p $WORKING_DIR
mkdir -p $OUTPUT_DIR
ls $INPUT_DIR/*.$EXT | parallel -j16 "python freq_count.py -f {} > $WORKING_DIR/{/.}.freq"
create_freq_timeseries.py -d $WORKING_DIR -s $STARTTIMEPOINT -e $ENDTIMEPOINT -p $STEP -f $WORKING_DIR/freq_timeseries.csv --log10
detect_changepoints_word_ts.py -f $WORKING_DIR/freq_timeseries.csv -v $FILTER_VOCAB_FILE -p $OUTPUT_DIR/pval_source_mean_bs_1000.csv -n $OUTPUT_DIR/sample_source_mean_bs_1000.csv -c $STARTTIMEPOINT -d
