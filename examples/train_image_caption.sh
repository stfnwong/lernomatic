#!/bin/bash
# Train image captioner

PYTHON=python

# Data
COCO_TRAIN_DATA="hdf5/coco-train.h5"
COCO_VAL_DATA="hdf5/coco-val.h5"
COCO_WORDMAP="hdf5/coco_wordmap.json"
# overfit data (for lr find)
COCO_OVERFIT_TRAIN="hdf5/coco-overfit-train.h5"
COCO_OVERFIT_VAL="hdf5/coco-overfit-val.h5"
# Training
INITIAL_BATCH_SIZE=48
FINE_TUNE_BATCH_SIZE=32
INITIAL_NUM_EPOCHS=20
# Device
DEVICE_ID=1

time $PYTHON examples/ex_train_image_capt.py \
    --verbose \
    --find-lr \
    --device-id=$DEVICE_ID \
    --train-data-path=$COCO_TRAIN_DATA \
    --val-data-path=$COCO_VAL_DATA \
    --overfit-train-data=$COCO_OVERFIT_TRAIN \
    --overfit-val-data=$COCO_OVERFIT_VAL \
    --wordmap=$COCO_WORDMAP \
    --batch-size=$INITIAL_BATCH_SIZE \
    --fine-tune-batch-size=$FINE_TUNE_BATCH_SIZE \
    --num-epochs=$INITIAL_NUM_EPOCHS


