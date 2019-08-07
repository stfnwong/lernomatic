#!/bin/bash
# Generate some images from a DCGAN model

EPOCH_NUM=79    # for output name
IMAGE_SIZE=64
#CHECKPOINT_FILE='checkpoint/celeba-dcgan-weight-init-128x128_iter_9000_epoch_34.pkl'
CHECKPOINT_FILE='checkpoint/celeba-dcgan-weight-init-64x64_iter_21000_epoch_79.pkl'
#CHECKPOINT_FILE='checkpoint/dcgan_celeba__iter_20100_epoch_79.pkl'
# device 
DEVICE_ID=-1
# Python 3.6 compatible interpreter name
PYTHON=python


for i in {1..64}
do
    IMG_OUTFILE="dcgan-64x64-epoch-$EPOCH_NUM-image-$i.png"
    $PYTHON examples/gan/ex_infer_dcgan.py\
        $CHECKPOINT_FILE \
        --device-id=$DEVICE_ID\
        --image-size=$IMAGE_SIZE\
        --img-outfile=$IMG_OUTFILE
done
