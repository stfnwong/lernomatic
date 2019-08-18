#!/bin/bash
# Generate HDF5 files for DCGAN unit tests 

# Name of Python 3.6 compatible interpreter
PYTHON=python

# Location of data 
DATASET_ROOT="/mnt/ml-data/datasets/celeba/"
EXTENSION="jpg"
# What size image to generate
IMAGE_SIZE=128
SPLIT_NAME='dcgan_unit_test'
# name of output dataset
DATASET_OUTFILE="hdf5/dcgan_unit_test.h5"
# Since this is just a unit test only get a relatively small 
# number of data points 
SIZE=4096


# Proces the data
$PYTHON tools/dataset/gan_image_proc.py \
    $DATASET_ROOT \
    --verbose \
    --outfile=$DATASET_OUTFILE \
    --image-size=$IMAGE_SIZE \
    --size=$SIZE \
    --extension=$EXTENSION \
    --split-name=$SPLIT_NAME
