INPUT_DIR=$1
WORKING_DIR=$2
OUTPUT_DIR=$3
STARTTIMEPOINT=$4
ENDTIMEPOINT=$5
STEP=$6
FILTER_VOCAB_FILE=$7
mkdir -p $WORKING_DIR
mkdir -p $OUTPUT_DIR

ls $INPUT_DIR/*.$EXT | parallel -j16 "python pos_tag.py -f {} -o $WORKING_DIR/{/.}.posdist"

pos_displacements.py -f $FILTER_VOCAB_FILE -d $WORKING_DIR/ -p "" -os pos -es ".posdist" -ps "" -sy $STARTTIMEPOINT -ey $ENDTIMEPOINT -s $STEP -e "pos" -o $WORKING_DIR

dump_timeseries.py -f $WORKING_DIR/timeseries_s_t_pos.pkl -s $WORKING_DIR/pos_source.csv -e $WORKING_DIR/pos_dest.csv -m $STARTTIMEPOINT -n $ENDTIMEPOINT -st $STEP -me "polar" -metric "jsd"

detect_changepoints_word_ts.py -f $WORKING_DIR/pos_source.csv -v $FILTER_VOCAB_FILE -p $OUTPUT_DIR/pval_source_mean_bs_1000.csv -n $OUTPUT_DIR/sample_source_mean_bs_1000.csv -c $STARTTIMEPOINT
