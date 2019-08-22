#!/bin/bash
# Generate HDF5 files for Night2Day dataset (pix2pix)

# Name of Python 3.6 compatible interpreter
PYTHON=python

# ======= TRAIN ======== #

EXTENSION="jpg"
# What size image to generate
IMAGE_SIZE=256
# split information
TRAIN_DATASET_ROOT="/mnt/ml-data/datasets/cyclegan/night2day/train/"
TRAIN_SPLIT_NAME='night2day-train'
TRAIN_DATASET_OUTFILE="hdf5/night2day-train-$IMAGE_SIZE.h5"
SIZE=0

# Process data
$PYTHON tools/gan/aligned_dataset_proc.py \
    --verbose \
    --dataset-root=$TRAIN_DATASET_ROOT \
    --mode=split \
    --outfile=$TRAIN_DATASET_OUTFILE \
    --image-size=$IMAGE_SIZE \
    --size=$SIZE \
    --extension=$EXTENSION \

# ======= TEST ======== #
# split information
TEST_DATASET_ROOT="/mnt/ml-data/datasets/cyclegan/night2day/test/"
TEST_SPLIT_NAME='night2day-test'
TEST_DATASET_OUTFILE="hdf5/night2day-test-$IMAGE_SIZE.h5"

$PYTHON tools/gan/aligned_dataset_proc.py \
    --verbose \
    --dataset-root=$TEST_DATASET_ROOT \
    --mode=split \
    --outfile=$TEST_DATASET_OUTFILE \
    --image-size=$IMAGE_SIZE \
    --size=$SIZE \
    --extension=$EXTENSION \


# ======= VAL ======== #
# split information
VAL_DATASET_ROOT="/mnt/ml-data/datasets/cyclegan/night2day/val/"
VAL_SPLIT_NAME='night2day-val'
VAL_DATASET_OUTFILE="hdf5/night2day-val-$IMAGE_SIZE.h5"

# Process
$PYTHON tools/gan/aligned_dataset_proc.py \
    --verbose \
    --dataset-root=$VAL_DATASET_ROOT \
    --mode=split \
    --outfile=$VAL_DATASET_OUTFILE \
    --image-size=$IMAGE_SIZE \
    --size=$SIZE \
    --extension=$EXTENSION \
