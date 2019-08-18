"""
EX_INFER_DCGAN
Infer on a DCGAN Model

Stefan Wong 2019
"""

import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from lernomatic.infer.gan import dcgan_inferrer
from lernomatic.util import image_util
from lernomatic.util import dcgan_util

GLOBAL_OPTS = dict()
TOOL_MODES = ('single', 'seed', 'history', 'walk')



# Get an inferrer object
def get_inferrer(device_id:int) -> dcgan_inferrer.DCGANInferrer:
    inferrer = dcgan_inferrer.DCGANInferrer(
        None,
        device_id = device_id
    )
    return inferrer


# Write image vector to disk
def write_img(fig, ax, fname:str, img_tensor:torch.Tensor) -> None:
    out_img = image_util.tensor_to_img(img_tensor)
    # get figures
    ax.imshow(out_img)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    fig.tight_layout()
    fig.savefig(fname)


# Generate a set of images from a noise vector
def generate_image() -> None:
    inferrer = get_inferrer(device_id = GLOBAL_OPTS['device_id'])
    inferrer.load_model(GLOBAL_OPTS['checkpoint_data'])
    fig, ax = plt.subplots()

    for n in range(GLOBAL_OPTS['num_images']):
        img = inferrer.forward()
        if GLOBAL_OPTS['num_images'] == 1:
            write_img(fig, ax, GLOBAL_OPTS['outfile'], img)
        else:
            path, ext = os.path.splitext(GLOBAL_OPTS['outfile'])
            fname = str(path) + '_' + str(n+1) + '.' + str(ext)
            write_img(fig, ax, fname, img)


# Generate one image from a random seed
def generate_from_seed() -> None:
    inferrer = get_inferrer(device_id = GLOBAL_OPTS['device_id'])
    inferrer.load_model(GLOBAL_OPTS['checkpoint_data'])

    fig, ax = plt.subplots()
    img = Image.open(GLOBAL_OPTS['seed_file']).convert('RGB')
    inp_tensor = image_util.img_to_tensor(img)
    out_tensor = inferrer.forward(inp_tensor)
    write_img(fig, ax, GLOBAL_OPTS['outfile'], out_tensor)


# Show the progression of the GAN outputs over the training cycl:w
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



# Walk along a series of random links
def walk() -> None:
    inferrer = get_inferrer(device_id = GLOBAL_OPTS['device_id'])
    inferrer.load_model(GLOBAL_OPTS['checkpoint_data'])

    p_prev = np.random.randn(inferrer.get_zvec_dim())
    p_next = np.random.randn(inferrer.get_zvec_dim())
    fig, ax = plt.subplots()
    img_idx = 1
    for link in range(GLOBAL_OPTS['num_links']):
        print('Walking link %d / %d' % (link+1, GLOBAL_OPTS['num_links']))

        out_imgs = walk_link(
            inferrer,
            p_prev,
            p_next,
            GLOBAL_OPTS['num_points']
        )

        # write files to disk
        for p, img in enumerate(tqdm(out_imgs, unit='images')):
            path, ext = os.path.splitext(GLOBAL_OPTS['outfile'])
            fname = str(path) + '_' + str(img_idx) + str(ext)
            write_img(fig, ax, fname, img)
            img_idx += 1

        p_prev = p_next
        p_next = np.random.randn(inferrer.get_zvec_dim())


# Walk between two points
def walk_link(inferrer:dcgan_inferrer.DCGANInferrer,
              p1:np.ndarray,
              p2:np.ndarray,
              num_points:int=32) -> None:
    points = dcgan_util.interp_walk(p1, p2, num_points)

    out_img_list = list()
    for p in range(num_points):
        x_t = torch.Tensor(points[p, :])
        x_t = x_t[None, :, None, None]
        out_img = inferrer.forward(x_t)
        out_img_list.append(out_img)

    return out_img_list


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
                        choices=TOOL_MODES,
                        help='Tool mode. (default: single)'
                        )
    parser.add_argument('--interp-mode',
                        type=str,
                        choices = ('linear', 'spherical'),
                        default='linear',
                        help='Interpolation to use in walk mode (default: linear)'
                        )
    parser.add_argument('--num-points',
                        type=int,
                        default=32,
                        help='In walk mode, number of points to interpolate (default: 32)'
                        )
    # TODO : for docstring - a link connects two points in a walk
    parser.add_argument('--num-links',
                        type=int,
                        default=1,
                        help='Number of links in the walk to do (default: 1)'
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
    parser.add_argument('--num-images',
                        type=int,
                        default=1,
                        help='Number of output files to generate (default: 1)'
                        )
    parser.add_argument('--outfile',
                        type=str,
                        default='figures/dcgan_output.png',
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
    elif GLOBAL_OPTS['mode'] == 'walk':
        walk()
    else:
        raise ValueError('Invalid tool mode [%s]' % str(GLOBAL_OPTS['mode']))
