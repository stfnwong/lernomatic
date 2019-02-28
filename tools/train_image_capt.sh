#!/bin/bash
# Script to run image caption training 

# data path options 
TRAIN_DATA_PATH="hdf5/coco_train_5000.h5"
TEST_DATA_PATH="hdf5/coco_test_5000.h5"
VAL_DATA_PATH="hdf5/coco_val_5000.h5"
WORDMAP_PATH="hdf5/wordmap_5000.json"
# other options 
DEVICE_ID=1
BATCH_SIZE=18
NUM_EPOCHS=2


time python examples/ex_train_image_capt.py \
    --verbose \
    --train-data-path=$TRAIN_DATA_PATH \
    --test-data-path=$TEST_DATA_PATH \
    --val-data-path=$VAL_DATA_PATH \
    --wordmap=$WORDMAP_PATH \
    --device-id=$DEVICE_ID \
    --batch-size=$BATCH_SIZE \
    --num-epochs=$NUM_EPOCHS
