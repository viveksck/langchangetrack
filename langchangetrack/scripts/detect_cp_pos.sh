INPUT_DIR=$1
WORKING_DIR=$2
OUTPUT_DIR=$3
STARTTIMEPOINT=$4
ENDTIMEPOINT=$5
STEP=$6
FILTER_VOCAB_FILE=$7
mkdir -p $WORKING_DIR
mkdir -p $OUTPUT_DIR

mkdir -p $WORKING_DIR/displacements/
pos_displacements.py -f $FILTER_VOCAB_FILE -d $INPUT_DIR/ -p "" -os pos -es ".posdist" -ps "" -sy $STARTTIMEPOINT -ey $ENDTIMEPOINT -s $STEP -e "pos" -o $WORKING_DIR/displacements

mkdir -p $WORKING_DIR/timeseries/
dump_timeseries.py -f $WORKING_DIR/displacements/timeseries_s_t_pos.pkl -s $WORKING_DIR/timeseries/source.csv -e $WORKING_DIR/timeseries/dest.csv -m $STARTTIMEPOINT -n $ENDTIMEPOINT -st $STEP -me "polar" -metric "jsd"

detect_changepoints_word_ts.py -f $WORKING_DIR/timeseries/source.csv -v $FILTER_VOCAB_FILE -p $OUTPUT_DIR/pvals.csv -n $OUTPUT_DIR/samples.csv -c $STARTTIMEPOINT
