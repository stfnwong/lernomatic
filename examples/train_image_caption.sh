#!/bin/bash
# Train image captioner

PYTHON=python

# Data
COCO_TRAIN_DATA="hdf5/coco-train.h5"
COCO_VAL_DATA="hdf5/coco-val.h5"
COCO_WORDMAP="hdf5/coco_wordmap.json"
# Training
INITIAL_BATCH_SIZE=64
FINE_TUNE_BATCH_SIZE=32
INITIAL_NUM_EPOCHS=20
# Device
DEVICE_ID=1

time $PYTHON examples/ex_train_image_capt.py \
    --verbose \
    --device-id=$DEVICE_ID \
    --train-data-path=$COCO_TRAIN_DATA \
    --test-data-path=$COCO_VAL_DATA \
    --wordmap=$COCO_WORDMAP \
    --batch-size=$INITIAL_BATCH_SIZE \
    --fine-tune-batch-size=$FINE_TUNE_BATCH_SIZE \
    --num-epochs=$INITIAL_NUM_EPOCHS


