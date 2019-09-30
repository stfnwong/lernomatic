"""
EX_TRAIN_LRGAN
Train an LRGAN

Stefan Wong 2019
"""

import argparse
# timing
import time
from datetime import timedelta
# visions
from torchvision import datasets
from torchvision import transforms
# library stuff
from lernomatic.data import hdf5_dataset
from lernomatic.models.gan import lrgan
from lernomatic.train.gan import lrgan_trainer
from lernomatic.train import schedule
from lernomatic.vis import vis_loss_history
from lernomatic.util import math_util
from lernomatic.util.gan import gan_util
# command line options
from lernomatic.options import options

# debug
#from pudb import set_trace; set_trace()

GLOBAL_OPTS = dict()



def main() -> None:
    if GLOBAL_OPTS['dataset'] is not None:
        train_dataset = hdf5_dataset.HDF5Dataset(
            GLOBAL_OPTS['dataset'],
            feature_name = 'images',
            label_name = 'labels',
        )
    else:
        gan_data_transform = transforms.Compose([
            transforms.Resize(GLOBAL_OPTS['image_size']),
            transforms.CenterCrop(GLOBAL_OPTS['image_size']),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        train_dataset = datasets.ImageFolder(
            root=GLOBAL_OPTS['dataset_root'],
            transform = gan_data_transform
        )

    # get some models
    generator = lrgan.LRGANGenerator(
        zvec_dim = GLOBAL_OPTS['zvec_dim'],
        num_filters = GLOBAL_OPTS['g_num_filters'],
        img_size = GLOBAL_OPTS['image_size']
    )
    discriminator = lrgan.LRGANDiscriminator(
        num_filters = GLOBAL_OPTS['d_num_filters'],
        img_size = GLOBAL_OPTS['image_size']
    )

    # init the weights
    gan_util.weight_init(generator)
    gan_util.weight_init(discriminator)



def get_parser() -> argparse.ArgumentParser:
    parser = options.get_trainer_options()
    # General opts
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        default=False,
                        help='Set verbose mode'
                        )
    # Network options
    parser.add_argument('--zvec-dim',
                        type=int,
                        default=128,
                        help='Dimension of z vector'
                        )
    parser.add_argument('--g-num-filters',
                        type=int,
                        default=64,
                        help='Number of filters to use in generator'
                        )
    parser.add_argument('--d-num-filters',
                        type=int,
                        default=64,
                        help='Number of filters to use in discriminator'
                        )
    # DCGAN trainer options
    parser.add_argument('--beta1',
                        type=float,
                        default=0.5,
                        help='beta1 parameter for ADAM optimizer'
                        )
    # Data options
    parser.add_argument('--image-size',
                        type=int,
                        default=64,
                        help='Resize all images to this size using a transformer before training'
                        )
    parser.add_argument('--dataset-root',
                        type=str,
                        default='/mnt/ml-data/datasets/celeba/',
                        help='Path to root of dataset'
                        )
    parser.add_argument('--dataset',
                        type=str,
                        default=None,
                        help='Path to dataset in HDF5 format (default: None)'
                        )
    # checkpoint options
    parser.add_argument('--checkpoint-dir',
                        type=str,
                        default='./checkpoint/',
                        help='Set directory to place training snapshots into'
                        )
    parser.add_argument('--checkpoint-name',
                        type=str,
                        default='dcgan',
                        help='Name to prepend to all checkpoints'
                        )
    parser.add_argument('--load-checkpoint',
                        type=str,
                        default=None,
                        help='Load a given checkpoint'
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
