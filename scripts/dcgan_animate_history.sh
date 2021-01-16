#!/bin/bash
# Script to animate history of DCGAN training 

# name of python interpreter 
PYTHON=python


# TODO : put in loop (to generate N animations from randomly selected 
# locations on the latent hypersphere
$PYTHON examples/gan/ex_infer_dcgan.py \
    dcgan_celeba_batch_128_history.txt \
    --verbose \
    --outfile=dcgan_batch128_120_epochs.png \
    --mode=history 

ffmpeg -i dcgan_batch128_120_epochs_%d.png -r 30 -framerate 2 -c:v libx264 dcgan_128_v3_face25.mp4
