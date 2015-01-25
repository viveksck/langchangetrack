INPUT_DIR=$1
WORKING_DIR=$2
OUTPUT_DIR=$3
STARTTIMEPOINT=$4
ENDTIMEPOINT=$5
STEP=$6
MODEL_FAMILY=$7
KNN=$8
NUMWORDS=$9
FILTER_VOCAB_FILE=${10}

EMBEDDINGS_TYPE=skipgram
echo "Output directory is", $OUTPUT_DIR

mkdir -p $WORKING_DIR
mkdir -p $OUTPUT_DIR

echo "Mapping to joint space"
mkdir -p $WORKING_DIR/predictors
echo "Predictors will be stored in", $WORKING_DIR/predictors
arr=("$INPUT_DIR/*.model")
((FINALTIMEPOINT=$ENDTIMEPOINT-$STEP))
parallel -j16 learn_map.py -k ${KNN} -f $WORKING_DIR/predictors/{/.}.predictor -o {} -n {//}/${FINALTIMEPOINT}_*.model -m $MODEL_FAMILY ::: $arr

echo "Creating words file"
WORDS_FILE=$WORKING_DIR/words.txt
((NUMLINES = NUMWORDS + 1)) 
cut -f 1 -d' ' $INPUT_DIR/${FINALTIMEPOINT}_*.model | head -n $NUMLINES > $WORDS_FILE
sed '1d' $WORDS_FILE | sponge $WORDS_FILE

echo "Computing displacements"
mkdir -p $WORKING_DIR/displacements/
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_DYNAMIC=FALSE
embedding_displacements.py -f $WORDS_FILE -d $INPUT_DIR/ -p $WORKING_DIR/predictors/ -os words -es ".model" -ps ".predictor" -sy $STARTTIMEPOINT -ey $ENDTIMEPOINT -s $STEP -e $EMBEDDINGS_TYPE -o $WORKING_DIR/displacements/

echo "Creating time series"
mkdir -p $WORKING_DIR/timeseries/
dump_timeseries.py -f $WORKING_DIR/displacements/timeseries_s_t_words.pkl -s $WORKING_DIR/timeseries/source.csv -e $WORKING_DIR/timeseries/dest.csv -m $STARTTIMEPOINT -n $ENDTIMEPOINT -st $STEP -me "polar" -metric "cosine"

detect_changepoints_word_ts.py -f $WORKING_DIR/timeseries/source.csv -v $FILTER_VOCAB_FILE -p $OUTPUT_DIR/pvals.csv -n $OUTPUT_DIR/samples.csv -c $STARTTIMEPOINT
