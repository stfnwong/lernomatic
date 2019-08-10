"""
EX_INFER_DCGAN
Infer on a DCGAN Model

Stefan Wong 2019
"""

import os
import torch
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from lernomatic.infer.gan import dcgan_inferrer
from lernomatic.util import image_util

GLOBAL_OPTS = dict()
TOOL_MODES = ('single', 'seed', 'history')


def get_inferrer(device_id:int) -> dcgan_inferrer.DCGANInferrer:
    inferrer = dcgan_inferrer.DCGANInferrer(
        None,
        device_id = device_id
    )
    return inferrer


def write_img(fig, ax, fname:str, img_tensor:torch.Tensor) -> None:
    out_img = image_util.tensor_to_img(img_tensor)
    # get figures
    ax.imshow(out_img)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    fig.tight_layout()
    fig.savefig(fname)


# NOTE: in effect this is using a new (unique) noise vector for each image.
# ANOTHER NOTE: we could just add a loop parameter to call this N times rather
# than use that janky shell script that is currently in the repo
def generate_image() -> None:
    inferrer = get_inferrer(device_id = GLOBAL_OPTS['device_id'])
    inferrer.load_model(GLOBAL_OPTS['checkpoint_data'])
    img = inferrer.forward()
    fig, ax = plt.subplots()
    write_img(fig, ax, GLOBAL_OPTS['outfile'], img)


def generate_from_seed() -> None:
    inferrer = get_inferrer(device_id = GLOBAL_OPTS['device_id'])
    inferrer.load_model(GLOBAL_OPTS['checkpoint_data'])

    img = Image.open(GLOBAL_OPTS['seed_file']).convert('RGB')
    inp_tensor = image_util.img_to_tensor(img)
    out_tensor = inferrer.forward(inp_tensor)
    fig, ax = plt.subplots()
    write_img(fig, ax, GLOBAL_OPTS['outfile'], out_tensor)


# TODO : first do history of a single image, then add option to do 8x8 grid of
# images, each one taken from some point in training history
def generate_history() -> None:
    # In this mode we interpret the checkpoint_data file as a text file that
    # contains a (newline-seperated) list of checkpoint filenames
    ck_files = []
    with open(GLOBAL_OPTS['checkpoint_data'], 'r') as fp:
        for line in fp:
            ck_files.append(line.strip('\n'))

    if GLOBAL_OPTS['verbose']:
        print('Read %d filenames from file [%s]' % (len(ck_files), str(GLOBAL_OPTS['checkpoint_data'])))

    inferrer = get_inferrer(GLOBAL_OPTS['device_id'])
    # get a random vector that we can re-use for each moment in history
    inferrer.load_model(ck_files[0])
    input_zvec = inferrer.get_random_zvec()

    out_img_set = []
    for n, ck in enumerate(ck_files):
        if GLOBAL_OPTS['verbose']:
            print('Generating image [%d / %d]' % (n+1, len(ck_files)), end='\r')
        inferrer.load_model(ck)
        out_img = inferrer.forward(input_zvec)
        out_img_set.append(out_img)

    if GLOBAL_OPTS['verbose']:
        print('\n done')

    fig, ax = plt.subplots()
    out_img_filenames = [
        os.path.splitext(GLOBAL_OPTS['outfile'])[0] \
        + '_' + str(n) for n in range(len(out_img_set))
    ]
    for n, (out_img, out_file) in enumerate(zip(out_img_set, out_img_filenames)):
        write_img(fig, ax, out_file, out_img)
        if GLOBAL_OPTS['verbose']:
            print('Wrote file [%3d / %3d] [%s]' % (n+1, len(out_img_set), str(out_file)), end='\r')

    print('\n done')


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
    # set tool mode
    parser.add_argument('--mode',
                        type=str,
                        default='single',
                        help='Tool mode. Must be one of %s (default: single)' % str(TOOL_MODES)
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
    parser.add_argument('--outfile',
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

    if GLOBAL_OPTS['mode'] == 'single':
        generate_image()
    elif GLOBAL_OPTS['mode'] == 'seed':
        generate_from_seed()
    elif GLOBAL_OPTS['mode'] == 'history':
        generate_history()
    else:
        raise ValueError('Invalid tool mode [%s]' % str(GLOBAL_OPTS['mode']))
