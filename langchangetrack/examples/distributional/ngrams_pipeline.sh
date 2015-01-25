INPUT_DIR=$1
WORKING_DIR=$2
OUTPUT_DIR=$3
EMBEDDINGS_TYPE=skipgram
EXT=ngrams
arr=("$INPUT_DIR/*.$EXT")
echo $arr
echo $OUTPUT_DIR
mkdir -p $WORKING_DIR
mkdir -p $OUTPUT_DIR
echo "Training embeddings"
parallel -vv -j 16 --progress train_embeddings_ngrams.py -f {} -o $WORKING_DIR -p {/.} -e $EMBEDDINGS_TYPE ::: $arr
