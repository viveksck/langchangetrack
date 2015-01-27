CORPUS_DIR=$1
WORKING_DIR=$2
OUTPUT_DIR=$3
STARTTIMEPOINT=$4
ENDTIMEPOINT=$5
STEP=$6
MODEL_FAMILY=$7
KNN=$8
NUMWORDS=$9
EXT=${10}
FILTER_VOCAB_FILE=${11}
WORKERS=${12}
EMBEDDINGS_TYPE=skipgram

arr=("$CORPUS_DIR/*.$EXT")
echo "Processing files", $arr
echo "Output directory is", $OUTPUT_DIR

mkdir -p $WORKING_DIR
mkdir -p $OUTPUT_DIR

echo "Training embeddings"
mkdir -p $WORKING_DIR/models
echo "Models will be stored in", $WORKING_DIR/models
parallel -vv -j ${WORKERS} --progress train_embeddings_ngrams.py -f {} -o $WORKING_DIR/models -p {/.} -e $EMBEDDINGS_TYPE -workers ${WORKERS} ::: $arr

detect_cp_distributional.sh $WORKING_DIR/models/ $WORKING_DIR $OUTPUT_DIR $STARTTIMEPOINT $ENDTIMEPOINT $STEP "locallinear" 100 1000 $FILTER_VOCAB_FILE ${WORKERS}
