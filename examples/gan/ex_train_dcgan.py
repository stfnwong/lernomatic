"""
EX_TRAIN_DCGAN
Train a DCGAN

Stefan Wong 2019
"""

import time
from datetime import timedelta
import argparse
from torchvision import datasets
from torchvision import transforms
from lernomatic.data import hdf5_dataset
from lernomatic.models.gan import dcgan_basic
from lernomatic.train.gan import dcgan_trainer
from lernomatic.vis import vis_loss_history

# debug
from pudb import set_trace; set_trace()

GLOBAL_OPTS = dict()


def main() -> None:
    if GLOBAL_OPTS['dataset'] is not None:
        celeba_transform = transforms.Compose([
            transforms.Resize(GLOBAL_OPTS['image_size']),
            transforms.CenterCrop(GLOBAL_OPTS['image_size']),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        train_dataset = hdf5_dataset.HDF5Dataset(
            GLOBAL_OPTS['dataset'],
            feature_name = 'images',
            label_name = 'labels',
            transform = celeba_transform
        )
    else:
        celeba_transform = transforms.Compose([
            transforms.Resize(GLOBAL_OPTS['image_size']),
            transforms.CenterCrop(GLOBAL_OPTS['image_size']),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        train_dataset = datasets.ImageFolder(
            root=GLOBAL_OPTS['dataset_root'],
            transform = celeba_transform
        )

    # get a model
    generator = dcgan_basic.DCGGenerator(
        zvec_dim = GLOBAL_OPTS['zvec_dim'],
        #num_filters = GLOBAL_OPTS['image_size']
        num_filters = GLOBAL_OPTS['g_num_filters'],
    )
    discriminator = dcgan_basic.DCGDiscriminator(
        #num_filters = GLOBAL_OPTS['image_size']
        num_filters = GLOBAL_OPTS['d_num_filters']
    )

    # get a trainer
    gan_trainer = dcgan_trainer.DCGANTrainer(
        discriminator,
        generator,
        #  DCGAN trainer specific arguments
        beta1 = GLOBAL_OPTS['beta1'],
        # general trainer arguments
        train_dataset = train_dataset,
        # training opts
        learning_rate   = GLOBAL_OPTS['learning_rate'],
        num_epochs      = GLOBAL_OPTS['num_epochs'],
        batch_size      = GLOBAL_OPTS['batch_size'],
        # Checkpoints
        save_every      = GLOBAL_OPTS['save_every'],
        checkpoint_name = GLOBAL_OPTS['checkpoint_name'],
        checkpoint_dir  = GLOBAL_OPTS['checkpoint_dir'],
        # display
        print_every     = GLOBAL_OPTS['print_every'],
        verbose         = GLOBAL_OPTS['verbose'],
        # device
        device_id       = GLOBAL_OPTS['device_id']
    )

    if GLOBAL_OPTS['load_checkpoint'] is not None:
        gan_trainer.load_checkpoint(GLOBAL_OPTS['load_checkpoint'])

    print(gan_trainer.device)
    train_start_time = time.time()
    gan_trainer.train()
    train_end_time = time.time()
    train_total_time = train_end_time - train_start_time
    print('Total training time : %s' % str(timedelta(seconds = train_total_time)))
    # show the training results
    dcgan_fig, dcgan_ax = vis_loss_history.get_figure_subplots(1)
    vis_loss_history.plot_train_history_dcgan(
        dcgan_ax,
        gan_trainer.get_g_loss_history(),
        gan_trainer.get_d_loss_history(),
        cur_epoch = gan_trainer.cur_epoch,
        iter_per_epoch = gan_trainer.iter_per_epoch
    )

    dcgan_fig.tight_layout()
    dcgan_fig.savefig('figures/dcgan_train_history.png')



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
                        default=100,
                        help='Print output every N epochs'
                        )
    parser.add_argument('--save-every',
                        type=int,
                        default=1000,
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
    parser.add_argument('--zvec-dim',
                        type=int,
                        default=100,
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
                        default=0.0002,
                        help='Learning rate for ADAM optimizer'
                        )
    parser.add_argument('--momentum',
                        type=float,
                        default=0.5,
                        help='Momentum for ADAM'
                        )
    parser.add_argument('--grad-clip',
                        type=float,
                        default=5.0,
                        help='Clip gradients at this (absolute) value'
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
                        default='dcgan_celeba_',
                        help='Name to prepend to all checkpoints'
                        )
    # TODO : need to implement this
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
