#!/bin/bash
# Generate HDF5 files for Night2Day dataset (pix2pix) 
# This generates a much smaller set of data for unit testing
# the Pix2PixTrainer module

# Name of Python 3.6 compatible interpreter
PYTHON=python

# ======= TRAIN ======== #

EXTENSION="jpg"
# What size image to generate
IMAGE_SIZE=256
# split information
UNITTEST_DATASET_ROOT="/mnt/ml-data/datasets/cyclegan/night2day/train/"
UNITTEST_SPLIT_NAME='night2day-unittest'
UNITTEST_DATASET_OUTFILE="hdf5/night2day-unittest-$IMAGE_SIZE.h5"
SIZE=4096

# Proces the data
$PYTHON tools/gan/aligned_dataset_proc.py \
    --verbose \
    --dataset-root=$UNITTEST_DATASET_ROOT \
    --mode=split \
    --outfile=$UNITTEST_DATASET_OUTFILE \
    --image-size=$IMAGE_SIZE \
    --size=$SIZE \
    --extension=$EXTENSION \

