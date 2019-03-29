#!/bin/bash
# Process all COCO data 

time python tools/proc_coco_data.py \
    --train-dataset-fname=hdf5/coco_train.h5 \
    --test-dataset-fname=hdf5/coco_test.h5 \
    --val-dataset-fname=hdf5/coco_val.h5 \
    --train-dataset-size=0 \
    --test-dataset-size=0 \
    --val-dataset-size=0 \
    --wordmap-fname=hdf5/wordmap_full.json \
    --coco-json=/mnt/ml-data/datasets/COCO/dataset_coco.json
