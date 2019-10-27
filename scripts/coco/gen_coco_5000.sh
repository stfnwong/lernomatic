#!/bin/bash
# Process the first 5000 elements of the COCO dataset

# Name of python 3.6 interpreter
PYTHON=python

time $PYTHON tools/proc_coco_data.py \
    --train-dataset-fname=hdf5/coco_train_5000.h5 \
    --test-dataset-fname=hdf5/coco_test_5000.h5 \
    --val-dataset-fname=hdf5/coco_val_5000.h5 \
    --train-dataset-size=5000 \
    --test-dataset-size=5000 \
    --val-dataset-size=5000 \
    --wordmap-fname=hdf5/wordmap_5000.json \
    --coco-json=/mnt/ml-data/datasets/COCO/dataset_coco.json
