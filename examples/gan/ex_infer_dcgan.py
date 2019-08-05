"""
EX_INFER_DCGAN
Infer on a DCGAN Model

Stefan Wong 2019
"""

import torch
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from lernomatic.infer.gan import dcgan_inferrer
from lernomatic.util import image_util

GLOBAL_OPTS = dict()


# TODO : does the image_size parameter have any effect in the inferrer module
# itself?

# NOTE: in effect this is using a new (unique) noise vector for each image.
def generate_image() -> None:
    inferrer = dcgan_inferrer.DCGANInferrer(
        None,
        img_size  = GLOBAL_OPTS['image_size'],
        device_id = GLOBAL_OPTS['device_id']
    )
    inferrer.load_model(GLOBAL_OPTS['checkpoint_data'])

    # TODO : generate a number of images to file
    img = inferrer.forward()
    out_img = image_util.tensor_to_img(img)
    # get figures
    fig, ax = plt.subplots()
    ax.imshow(out_img)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    fig.tight_layout()
    fig.savefig(GLOBAL_OPTS['img_outfile'])


def generate_from_seed() -> None:
    inferrer = dcgan_inferrer.DCGANInferrer(
        None,
        img_size  = GLOBAL_OPTS['image_size'],
        device_id = GLOBAL_OPTS['device_id']
    )
    inferrer.load_model(GLOBAL_OPTS['checkpoint_data'])

    img = Image.open(GLOBAL_OPTS['seed_file']).convert('RGB')
    inp_tensor = image_util.img_to_tensor(img)
    out_tensor = inferrer.forward(inp_tensor)

    out_img = image_util.tensor_to_img(out_tensor)
    fig, ax = plt.subplots()
    ax.imshow(out_img)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    fig.tight_layout()
    fig.savefig(GLOBAL_OPTS['img_outfile'])


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    # General opts
    parser.add_argument('checkpoint_data',
                        type=str,
                        help='Path to checkpoint data'
                        )
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        default=False,
                        help='Set verbose mode'
                        )
    # Device options
    parser.add_argument('--device-id',
                        type=int,
                        default=-1,
                        help='Set device id (-1 for CPU)'
                        )
    parser.add_argument('--batch-size',
                        type=int,
                        default=64,
                        help='Batch size to use during training'
                        )
    # Data options
    parser.add_argument('--image-size',
                        type=int,
                        default=64,
                        help='Resize all images to this size using a transformer before training'
                        )
    parser.add_argument('--seed-file',
                        type=str,
                        default=None,
                        help='Use this image as a seed rather than generating a random input vector (default: None)'
                        )
    # Output options
    parser.add_argument('--num-files',
                        type=int,
                        default=1,
                        help='Number of output files to generate (default: 1)'
                        )
    parser.add_argument('--img-outfile',
                        type=str,
                        default='figures/dcgan_celeba_output.png',
                        help='Name of output image'
                        )

    return parser


if __name__ == '__main__':
    parser = get_parser()
    opts   = vars(parser.parse_args())

    for k, v in opts.items():
        GLOBAL_OPTS[k] = v

    if GLOBAL_OPTS['verbose'] is True:
        print(' ---- GLOBAL OPTIONS ---- ')
        for k,v in GLOBAL_OPTS.items():
            print('\t[%s] : %s' % (str(k), str(v)))

    if GLOBAL_OPTS['seed_file'] is not None:
        generate_from_seed()
    else:
        generate_image()
