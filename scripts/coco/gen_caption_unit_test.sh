#!/bin/bash
# Create unit test data for image caption trainer
DATASET_SIZE=1000
DATASET_ROOT="/mnt/ml-data/datasets/COCO"
TRAIN_DATASET_NAME="hdf5/caption_unit_test_train.h5"
TEST_DATASET_NAME="hdf5/caption_unit_test_test.h5"
VAL_DATASET_NAME="hdf5/caption_unit_test_val.h5"
WORDMAP="hdf5/caption_unit_test_wordmap.json"
COCO_JSON_PATH="$DATASET_ROOT/dataset_coco.json"

# name of Python 3.6 compatible interpreter
PYTHON=python

time $PYTHON tools/proc_coco_data.py \
    --train-dataset-fname=$TRAIN_DATASET_NAME \
    --test-dataset-fname=$TEST_DATASET_NAME \
    --val-dataset-fname=$VAL_DATASET_NAME \
    --train-dataset-size=$DATASET_SIZE \
    --test-dataset-size=$DATASET_SIZE \
    --val-dataset-size=$DATASET_SIZE \
    --wordmap-fname=$WORDMAP \
    --coco-json=$COCO_JSON_PATH
