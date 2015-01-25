INPUT_DIR=$1
WORKING_DIR=$2
OUTPUT_DIR=$3
STARTTIME=$4
ENDTIME=$5
STEP=$6
EXT=ngrams
mkdir -p $WORKING_DIR
mkdir -p $OUTPUT_DIR
ls $INPUT_DIR/*.$EXT | parallel -j16 "python freq_count.py -f {} > $WORKING_DIR/{/.}.freq"
create_freq_timeseries.py -d $WORKING_DIR -s $STARTTIME -e $ENDTIME -p $STEP -f $OUTPUT_DIR/freq_timeseries.csv --log10
