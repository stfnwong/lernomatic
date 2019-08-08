"""
EX_INFER_PIX2PIX
Example of inference on a pix2pix model

Stefan Wong 2019
"""

import os
import argparse
from PIL import Image       # would be nice to have cv2 here...
import matplotlib.pyplot as plt
import torch
from torchvision import datasets
from torch.utils.data import Dataset
from lernomatic.infer.gan import pix2pix_inferrer
from lernomatic.models import common
from lernomatic.util import image_util

GLOBAL_OPTS = dict()


def translate_dataset(max_images:int=128) -> None:
    inferrer = pix2pix_inferrer.Pix2PixInferrer(
        device_id = GLOBAL_OPTS['device_id']
    )
    inferrer.load_model(GLOBAL_OPTS['checkpoint'])

    # Just assume for now that an ImageFolder dataset is fine
    dataset = datasets.ImageFolder(
        GLOBAL_OPTS['dataset_path'],
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size = 1,
        shuffle = False
    )

    # NOTE : for now, we just use the batch index as part of the filename
    for batch_idx, (data_a, data_b) in enumerate(data_loader):
        print('Processing image [%d / %d]' % (batch_idx+1, len(data_loader)), end='\r')
        fake_a = inferrer.forward(data_a)
        out_img = image_util.tensor_to_img(fake_a)

        fig_filename = 'figures/pix2pix/pix2pix_image_%d.png' % (str(batch_idx))
        fig, ax = plt.subplots()        # Is it faster to move this out of the loop?
        ax.imshow(out_img)
        fig.tight_layout()
        fig.savefig(fig_filename)

        if batch_idx >= max_images:
            break

    print('\n done')


def translate_image() -> None:
    inferrer = pix2pix_inferrer.Pix2PixInferrer(
        device_id = GLOBAL_OPTS['device_id']
    )
    inferrer.load_model(GLOBAL_OPTS['checkpoint'])

    img = Image.open(GLOBAL_OPTS['input_file']).convert('RGB')
    img_tensor = image_util.img_to_tensor(img)
    out_tensor = inferrer.forward(img_tensor)

    # convert back to an image for display
    out_img = image_util.tensor_to_img(out_tensor)

    fig, ax = plt.subplots()
    ax.imshow(out_img)
    fig.tight_layout()
    fig.savefig(GLOBAL_OPTS['img_outfile'])


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    # General opts
    parser.add_argument('checkpoint',
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

    # Data options
    parser.add_argument('--dataset-path',
                        type=str,
                        default=None,
                        help='Path to an entire dataset to evaluate (default: None)'
                        )
    parser.add_argument('--input-file',
                        type=str,
                        default=None,
                        help='Path to a single file to evaluate (default: None)'
                        )
    # Output options
    parser.add_argument('--img-outfile',
                        type=str,
                        default='pix2pix_out.png',
                        help='Name of output file for a single image'
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

    if GLOBAL_OPTS['dataset_path'] is not None:
        translate_dataset()
    elif GLOBAL_OPTS['input_file'] is not None:
        translate_image()
    else:
        raise ValueError('No --dataset-path or --input-file specified')
