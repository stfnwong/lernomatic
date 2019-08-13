"""
EX_TRAIN_PIX2PIX
Example which trains a pix2pix model

Stefan Wong 2019
"""

import os
import argparse
from lernomatic.train.gan import pix2pix_trainer
from lernomatic.models.gan.cycle_gan import pixel_disc
from lernomatic.models.gan.cycle_gan import nlayer_disc
from lernomatic.models.gan.cycle_gan import resnet_gen
from lernomatic.models.gan.cycle_gan import unet_gen
from lernomatic.models import common

from lernomatic.data.gan import aligned_dataset
from lernomatic.data.gan import gan_transforms

# measure training time
import time
from datetime import timedelta

# debug
#from pudb import set_trace; set_trace()


GLOBAL_OPTS = dict()


def get_dataset(ab_path:str, dataset_name:str, data_root:str, transforms=None) -> aligned_dataset.AlignedDatasetHDF5:

    # TODO : support more image sizes?
    if transforms is None:
        transforms = gan_transforms.get_gan_transforms(
            do_crop = True,
            to_tensor = True,
            do_scale_width = True
        )
    dataset = aligned_dataset.AlignedDataset(
        ab_path,
        data_root = data_root,
        transform = transforms
    )

    return dataset


# TODO: add in the rest of the kwargs - accept the defaults for now
def get_generator(gen_type:str, input_nc:int=3, output_nc:int=3, img_size:int=256) -> common.LernomaticModel:
    if gen_type == 'resnet':
        gen = resnet_gen.ResnetGenerator(
            input_nc,
            output_nc,
            num_filters = 64
        )
    elif gen_type == 'unet':
        if img_size == 128:
            num_downsamples = 7
        elif img_size == 256:
            num_downsamples = 8
        else:
            raise ValueError('img_size [%d] must be either 128, 256' % int(img_size))

        gen = unet_gen.UNETGenerator(
            input_nc,
            output_nc,
            num_downsamples,
            num_filters = 64
        )
    else:
        raise ValueError('Generator [%s] not supported' % str(gen_type))

    return gen


# TODO: add in the rest of the kwargs
def get_discriminator(disc_type:str, input_nc:int, num_filters:int=64, num_layers:int=3) -> common.LernomaticModel:
    if disc_type == 'nlayer':
        disc = nlayer_disc.NLayerDiscriminator(
            input_nc,
            num_filters = num_filters,
            num_layers = num_layers
        )
    elif disc_type == 'pixel':      # 1x1 PixelGAN discriminator
        disc = pixel_disc.PixelDiscriminator(
            input_nc,
            num_filters
        )
    else:
        raise ValueError('Discriminator [%s] not supported' % str(disc_type))

    return disc


def main() -> None:
    # get some models
    generator = get_generator(
        GLOBAL_OPTS['gen_type'],
        3,      # input channels
        3       # output channels
    )
    if GLOBAL_OPTS['verbose']:
        print('Got generator [%s]' % repr(generator))

    discriminator = get_discriminator(
        GLOBAL_OPTS['disc_type'],
        3 + 3,      # input channels + output_channels (when replacing with vars use input_nc + output_nc here)
    )
    if GLOBAL_OPTS['verbose']:
        print('Got discriminator [%s]' % repr(discriminator))

    # get paths
    train_ab_paths = [path for path in os.listdir(GLOBAL_OPTS['train_data_path'])]
    val_ab_paths = [path for path in os.listdir(GLOBAL_OPTS['val_data_path'])]

    # get some data
    train_dataset = get_dataset(
        train_ab_paths,
        'pix2xpix_train',
        GLOBAL_OPTS['train_data_path']
    )
    val_dataset   = get_dataset(
        val_ab_paths,
        'pix2xpix_val',
        GLOBAL_OPTS['val_data_path']
    )

    # get a trainer
    if GLOBAL_OPTS['load_checkpoint'] is not None:
        trainer = pix2pix_trainer.Pix2PixTrainer(
            None,
            None,
            # dataset
            train_dataset = train_dataset,
            val_dataset   = val_dataset,
            # checkpoint
            checkpoint_dir = GLOBAL_OPTS['checkpoint_dir'],
            checkpoint_name = GLOBAL_OPTS['checkpoint_name'],
            # display,
            print_every = GLOBAL_OPTS['print_every'],
            save_every = GLOBAL_OPTS['save_every'],
        )
        trainer.load_checkpoint(GLOBAL_OPTS['load_checkpoint'])
    else:
        trainer = pix2pix_trainer.Pix2PixTrainer(
            # models
            generator,
            discriminator,
            # dataset
            train_dataset = train_dataset,
            val_dataset   = val_dataset,
            # training params
            #batch_size    = GLOBAL_OPTS['batch_size'],
            batch_size    = 1,
            learning_rate = GLOBAL_OPTS['learning_rate'],
            num_epochs    = GLOBAL_OPTS['num_epochs'],
            # device
            device_id = GLOBAL_OPTS['device_id'],
            # checkpoint
            checkpoint_dir = GLOBAL_OPTS['checkpoint_dir'],
            checkpoint_name = GLOBAL_OPTS['checkpoint_name'],
            # display,
            print_every = GLOBAL_OPTS['print_every'],
            save_every = GLOBAL_OPTS['save_every'],
        )

    train_start_time = time.time()
    trainer.train()
    train_end_time = time.time()
    train_total_time = train_end_time - train_start_time
    print('Total training time : %s' % \
          str(timedelta(seconds=train_total_time))
    )



def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    # General opts
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        default=False,
                        help='Set verbose mode'
                        )
    parser.add_argument('--print-every',
                        type=int,
                        default=10,
                        help='Print output every N epochs'
                        )
    parser.add_argument('--save-every',
                        type=int,
                        default=-1,
                        help='Save model checkpoint every N epochs'
                        )
    parser.add_argument('--num-workers',
                        type=int,
                        default=1,
                        help='Number of workers to use when generating HDF5 files'
                        )
    # Device options
    parser.add_argument('--device-id',
                        type=int,
                        default=-1,
                        help='Set device id (-1 for CPU)'
                        )
    # Network options
    parser.add_argument('--gen-type',
                        type=str,
                        default='resnet',
                        help='Type of generator to use (resnet or unet)'
                        )
    parser.add_argument('--disc-type',
                        type=str,
                        default='pixel',
                        help='Type of discriminator to use (nlayer or pixel)'
                        )


    # Training options
    parser.add_argument('--start-epoch',
                        type=int,
                        default=0,
                        help='Epoch to start training from'
                        )
    parser.add_argument('--num-epochs',
                        type=int,
                        default=20,
                        help='Epoch to stop training at'
                        )
    parser.add_argument('--batch-size',
                        type=int,
                        default=64,
                        help='Batch size to use during training'
                        )
    parser.add_argument('--weight-decay',
                        type=float,
                        default=1e-4,
                        help='Weight decay to use for optimizer'
                        )
    parser.add_argument('--learning-rate',
                        type=float,
                        default=1e-3,
                        help='Learning rate for optimizer'
                        )
    # Data options
    parser.add_argument('--train-data-path',
                        type=str,
                        default='/home/kreshnik/ml-data/night2day/train/',
                        help='Path to training data'
                        )
    parser.add_argument('--val-data-path',
                        type=str,
                        default='/home/kreshnik/ml-data/night2day/val/',
                        help='Path to training data'
                        )
    parser.add_argument('--checkpoint-dir',
                        type=str,
                        default='./checkpoint',
                        help='Set directory to place training snapshots into'
                        )
    parser.add_argument('--checkpoint-name',
                        type=str,
                        default='pix2pix',
                        help='Name to prepend to all checkpoints'
                        )
    parser.add_argument('--load-checkpoint',
                        type=str,
                        default=None,
                        help='Load a given checkpoint'
                        )
    parser.add_argument('--overwrite',
                        action='store_true',
                        default=False,
                        help='Overwrite existing processed data files'
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
