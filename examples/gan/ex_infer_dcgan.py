"""
EX_INFER_DCGAN
Infer on a DCGAN Model

Stefan Wong 2019
"""

import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from lernomatic.infer.dcgan import dcgan_inferrer

GLOBAL_OPTS = dict()


def tensor_to_img(X:torch.Tensor) -> None:
    img = X.cpu().numpy()

    # get the image in a form suitable for display
    img = img.transpose(1, 2, 0)
    img_min = img.min()
    img = img + np.abs(img_min)     # get all values positive
    img_max = img.max()
    img = img / np.abs(img_max)     # normalize to [0..1]

    return img


def main() -> None:
    inferrer = dcgan_inferrer.DCGANInferrer(
        None,
        img_size  = GLOBAL_OPTS['image_size'],
        device_id = GLOBAL_OPTS['device_id']
    )
    inferrer.load_model(GLOBAL_OPTS['checkpoint_data'])

    # TODO : generate a number of images to file
    img = inferrer.forward()
    out_img = tensor_to_img(img)
    # get figures
    fig, ax = plt.subplots()
    ax.imshow(out_img)
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

    main()
