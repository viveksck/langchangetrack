INPUT_DIR=$1
WORKING_DIR=$2
OUTPUT_DIR=$3
STARTTIMEPOINT=$4
ENDTIMEPOINT=$5
STEP=$6
FILTER_VOCAB_FILE=$7

mkdir -p $WORKING_DIR
mkdir -p $OUTPUT_DIR
mkdir -p $WORKING_DIR/timeseries

create_freq_timeseries.py -d $INPUT_DIR -s $STARTTIMEPOINT -e $ENDTIMEPOINT -p $STEP -f $WORKING_DIR/timeseries/freq_timeseries.csv --log10

detect_changepoints_word_ts.py -f $WORKING_DIR/timeseries/freq_timeseries.csv -v $FILTER_VOCAB_FILE -p $OUTPUT_DIR/pvals.csv -n $OUTPUT_DIR/samples.csv -c $STARTTIMEPOINT -d
