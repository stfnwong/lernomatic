#!/bin/bash
# Generate Caltech birds dataset for use with GANs, VAEs, etc
# Stefan Wong 2019

# Name of Python 3.6 compatible interpreter
PYTHON=python

# Location of data 
DATASET_ROOT="/mnt/ml-data/datasets/CUB_200_2011/images/"
EXTENSION="jpg"
# What size image to generate
IMAGE_SIZE=128
SPLIT_NAME='cub_2011'
# name of output dataset
DATASET_OUTFILE="hdf5/cub_2011-$IMAGE_SIZE.h5"
SIZE=0

# Proces the data
$PYTHON tools/dataset/gan_image_proc.py \
    $DATASET_ROOT \
    --verbose \
    --outfile=$DATASET_OUTFILE \
    --image-size=$IMAGE_SIZE \
    --size=$SIZE \
    --extension=$EXTENSION \
    --split-name=$SPLIT_NAME
