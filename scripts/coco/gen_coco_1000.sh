#!/bin/bash
# Process the first 1000 elements of the COCO dataset

time python tools/proc_coco_data.py \
    --train-dataset-fname=hdf5/coco-train-1000.h5 \
    --test-dataset-fname=hdf5/coco-test-1000.h5 \
    --val-dataset-fname=hdf5/coco-val-1000.h5 \
    --train-dataset-size=1000 \
    --test-dataset-size=1000 \
    --val-dataset-size=1000 \
    --wordmap-fname=hdf5/wordmap-1000.json \
    --coco-json=/mnt/ml-data/datasets/COCO/dataset_coco.json
