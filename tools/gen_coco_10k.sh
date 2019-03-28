#!/bin/bash
# Process the first 10k elements of the COCO dataset

time python tools/proc_coco_data.py \
    --train-dataset-fname=hdf5/coco_train_10k.h5 \
    --test-dataset-fname=hdf5/coco_test_10k.h5 \
    --val-dataset-fname=hdf5/coco_val_10k.h5 \
    --train-dataset-size=10000 \
    --test-dataset-size=10000 \
    --val-dataset-size=10000 \
    --wordmap-fname=hdf5/wordmap_10k.json \
    --coco-json=/mnt/ml-data/datasets/COCO/dataset_coco.json
