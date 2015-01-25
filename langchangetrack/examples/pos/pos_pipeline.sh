INPUT_DIR=$1
WORKING_DIR=$2
OUTPUT_DIR=$3
STARTTIME=$4
ENDTIME=$5
STEP=$6
EXT=ngrams
mkdir -p $WORKING_DIR
mkdir -p $OUTPUT_DIR
ls $INPUT_DIR/*.$EXT | parallel -j16 "python pos_tag.py -f {} -o $WORKING_DIR/{/.}.posdist"
