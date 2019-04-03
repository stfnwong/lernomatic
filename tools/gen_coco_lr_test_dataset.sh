#!/bin/bash
# Create a smalle version of the test dataset for finding learning rates

TEST_DATASET_SIZE=5000
TEST_DATASET_NAME="hdf5/coco_lr_find_$TEST_DATSET_SIZE.h5"
WORDMAP="hdf5/lr_find_wordmap.json"
COCO_JSON="/mnt/ml-data/datasets/COCO/dataset_coco.json"

time python tools/proc_coco_data.py \
    --split=test \
    --test-dataset-fname=$TEST_DATASET_NAME \
    --test-dataset-size=$TEST_DATASET_SIZE \
    --wordmap-fname=$WORDMAP \
    --coco-json=$COCO_JSON
