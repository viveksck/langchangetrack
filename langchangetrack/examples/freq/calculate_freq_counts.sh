#!/bin/bash
CORPUS_DIR=$1
WORKING_DIR=$2
EXT=$3
WORKERS=$4
mkdir -p $WORKING_DIR
mkdir -p $WORKING_DIR/counts/
ls $CORPUS_DIR/*.$EXT | parallel -j${WORKERS} "freq_count.py -f {} > $WORKING_DIR/counts/{/.}.freq"
