#!/bin/bash
# Process all COCO data 

PYTHON=python
WORDMAP='coco_wordmap.json'
OVERFIT_SIZE=500
OVERFIT_NAME='overfit'
MAX_CAPT_LEN=64
OUTPUT_PREFIX="hdf5"
COCO_JSON="/mnt/ml-data/datasets/COCO/dataset_coco.json"

time $PYTHON tools/proc_coco_data.py \
    --train-dataset-fname="$OUTPUT_PREFIX/coco-train.h5" \
    --test-dataset-fname="$OUTPUT_PREFIX/coco-test.h5" \
    --val-dataset-fname="$OUTPUT_PREFIX/coco-val.h5" \
    --train-dataset-size=0 \
    --test-dataset-size=0 \
    --val-dataset-size=0 \
    --max-capt-len=$MAX_CAPT_LEN \
    --wordmap-fname="$OUTPUT_PREFIX/$WORDMAP" \
    --coco-json=$COCO_JSON

echo "Generating overfit dataset with $OVERFIT_SIZE elements"

OVERFIT_TRAIN_NAME="hdf5/coco-$OVERFIT_NAME-train.h5"
OVERFIT_TEST_NAME="hdf5/coco-$OVERFIT_NAME-test.h5"
OVERFIT_VAL_NAME="hdf5/coco-$OVERFIT_NAME-val.h5"
time $PYTHON tools/proc_coco_data.py \
    --train-dataset-fname=$OVERFIT_TRAIN_NAME \
    --test-dataset-fname=$OVERFIT_TEST_NAME \
    --val-dataset-fname=$OVERFIT_VAL_NAME \
    --max-capt-len=$MAX_CAPT_LEN \
    --wordmap-fname="$OUTPUT_PREFIX/overfit-$WORDMAP" \
    --train-dataset-size=$OVERFIT_SIZE \
    --test-dataset-size=$OVERFIT_SIZE \
    --val-dataset-size=$OVERFIT_SIZE \
    --coco-json=$COCO_JSON
