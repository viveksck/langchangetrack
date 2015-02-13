#!/bin/bash
CORPUS_DIR=$1
WORKING_DIR=$2
EXT=$3
WORKERS=$4
mkdir -p $WORKING_DIR
mkdir -p $WORKING_DIR/posdist/
ls $CORPUS_DIR/*.$EXT | parallel -j${WORKERS} "python pos_tag.py -f {} -o $WORKING_DIR/posdist/{/.}.posdist"
