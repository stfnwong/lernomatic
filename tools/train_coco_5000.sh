#!/bin/bash
# Train a caption generator on the small COCO dataset 
#
# Stefan Wong 2019

TRAIN_DATA_PATH='hdf5/coco_train_5000.h5'
TEST_DATA_PATH='hdf5/coco_test_5000.h5'
WORDMAP='hdf5/wordmap.json'
DEVICE_ID=1
NUM_EPOCHS=50
BATCH_SIZE=24

SAVE_EVERY=-1
PRINT_EVERY=50

time python examples/ex_train_image_capt.py \
    --verbose \
    --train-data-path=$TRAIN_DATA_PATH \
    --test-data-path=$TEST_DATA_PATH \
    --wordmap=$WORDMAP \
    --device-id=$DEVICE_ID \
    --num-epochs=$NUM_EPOCHS \
    --batch-size=$BATCH_SIZE \
    --save-every=$SAVE_EVERY \
    --print-every=$PRINT_EVERY
