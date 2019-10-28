#!/bin/bash
# Create a smalle version of the test dataset for finding learning rates

# Name of python 3.6 compatible interpreter
PYTHON=python

DATASET_ROOT="/mnt/ml-data/datasets/COCO"
DATASET_SIZE=5000
TRAIN_DATASET_NAME="hdf5/coco_lr_find_train_$DATASET_SIZE.h5"
TEST_DATASET_NAME="hdf5/coco_lr_find_test_$DATASET_SIZE.h5"
WORDMAP="hdf5/lr_find_wordmap.json"
COCO_JSON="$DATASET_ROOT/dataset_coco.json"


# train split 
time $PYTHON tools/proc_coco_data.py \
    --split=train \
    --test-dataset-fname=$TRAIN_DATASET_NAME \
    --train-dataset-size=$DATASET_SIZE \
    --wordmap-fname=$WORDMAP \
    --coco-json=$COCO_JSON

# Test split
time $PYTHON tools/proc_coco_data.py \
    --split=test \
    --test-dataset-fname=$TEST_DATASET_NAME \
    --test-dataset-size=$DATASET_SIZE \
    --wordmap-fname=$WORDMAP \
    --coco-json=$COCO_JSON
