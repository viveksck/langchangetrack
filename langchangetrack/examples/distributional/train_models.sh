#!/bin/bash
CORPUS_DIR=$1
WORKING_DIR=$2
EXT=${3}
WORKERS=${4}
EMBEDDINGS_TYPE=skipgram
EPOCHS=3
arr=("$CORPUS_DIR/*.$EXT")
echo "Processing files", $arr
echo "Training embeddings"
mkdir -p $WORKING_DIR/models
echo "Models will be stored in", $WORKING_DIR/models
parallel -vv -j ${WORKERS} --progress train_embeddings_ngrams.py -f {} -o $WORKING_DIR/models -p {/.} -e $EMBEDDINGS_TYPE -workers ${WORKERS} --epochs $EPOCHS ::: $arr
