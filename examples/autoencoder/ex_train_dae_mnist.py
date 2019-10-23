"""
EX_TRAIN_DAE
Train a de-noising autoencoder

Stefan Wong 2019
"""

import torch
import torchvision
import argparse
import numpy as np
import matplotlib.pyplot as plt
# timing
import time
from datetime import timedelta

from lernomatic.infer.autoencoder import dae_inferrer
from lernomatic.train.autoencoder import dae_trainer
from lernomatic.models.autoencoder import denoise_ae
from lernomatic.models import common        # mainly for type hints
# command line options
from lernomatic.options import options
# vis options
from lernomatic.vis import vis_loss_history
from lernomatic.vis import vis_img
from lernomatic.util import image_util


GLOBAL_OPTS = dict()


# Plotter function. For now I won't bother to generalize this, but eventually
# it might find its way in lernomatic.vis
def plot_denoise(
    ax,
    image_batch:torch.Tensor) -> None:

    if len(ax) != image_batch.shape[0]:
        raise ValueError('Number of image axes (%d) does not equal batch size (%d)' %\
                         (len(ax), image_batch.shape[0])
        )

    for img_idx in range(image_batch.shape[0]):
        img = image_util.tensor_to_img(image_batch[img_idx, :, :, :])
        ax[img_idx].imshow(img)
        ax[img_idx].get_xaxis().set_visible(False)
        ax[img_idx].get_yaxis().set_visible(False)



def get_datasets(data_dir:str) -> tuple:
    dataset_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize( (0.1307,), (0.3081,))
    ])

    train_dataset = torchvision.datasets.MNIST(
        data_dir,
        train = True,
        download = True,
        transform = dataset_transform
    )
    val_dataset = torchvision.datasets.MNIST(
        data_dir,
        train = False,
        download = True,
        transform = dataset_transform
    )

    return (train_dataset, val_dataset)


def main() -> None:
    train_dataset, val_dataset = get_datasets(GLOBAL_OPTS['data_dir'])

    # get some models
    encoder = denoise_ae.DAEEncoder(num_channels = 1)
    decoder = denoise_ae.DAEDecoder(num_channels = 1)

    trainer = dae_trainer.DAETrainer(
        encoder,
        decoder,
        # datasets
        train_dataset   = train_dataset,
        val_dataset     = val_dataset,
        device_id       = GLOBAL_OPTS['device_id'],
        # trainer params
        batch_size      = GLOBAL_OPTS['batch_size'],
        num_epochs      = GLOBAL_OPTS['num_epochs'],
        learning_rate   = GLOBAL_OPTS['learning_rate'],
        # checkpoints, saving, etc
        checkpoint_dir  = GLOBAL_OPTS['checkpoint_dir'],
        checkpoint_name = GLOBAL_OPTS['checkpoint_name'],
        save_every      = GLOBAL_OPTS['save_every'],
        print_every     = GLOBAL_OPTS['print_every'],
        verbose         = GLOBAL_OPTS['verbose']
    )

    # Add a summary writer
    if GLOBAL_OPTS['tensorboard_dir'] is not None:
        if GLOBAL_OPTS['verbose']:
            print('Adding tensorboard writer to [%s]' % repr(trainer))
        writer=tensorboard.SummaryWriter(log_dir=GLOBAL_OPTS['tensorboard_dir'])
        trainer.set_tb_writer(writer)

    # train the model
    train_start_time = time.time()
    trainer.train()
    train_end_time = time.time()
    train_total_time = train_end_time - train_start_time

    print('Trained [%s] for %d epochs, total time : %s' %\
          (repr(trainer), trainer.cur_epoch, str(timedelta(seconds = train_total_time)))
    )

    # Take the models and load them into an inferrer
    if GLOBAL_OPTS['infer']:
        trainer.drop_last = True
        # make the number of squares in the output figure settable later
        subplot_square = 8
        trainer.set_batch_size(subplot_square * subplot_square)      # So that we have an NxN grid of outputs
        # Get some figure stuff
        out_fig, out_ax_list     = vis_img.get_grid_subplots(subplot_square)
        noise_fig, noise_ax_list = vis_img.get_grid_subplots(subplot_square)

        # Get an inferrer
        inferrer = dae_inferrer.DAEInferrer(
            trainer.encoder,
            trainer.decoder,
            noise_bias   = GLOBAL_OPTS['noise_bias'],
            noise_factor = GLOBAL_OPTS['noise_factor'],
            device_id    = GLOBAL_OPTS['device_id']
        )
        if GLOBAL_OPTS['verbose']:
            print('Created [%s] object and attached models [%s], [%s]' %\
                  (repr(trainer), repr(trainer.encoder), repr(trainer.decoder))
            )

        infer_start_time = time.time()
        for batch_idx, (data, _) in enumerate(trainer.val_loader):
            print('Inferring batch [%d / %d]' % (batch_idx+1, len(trainer.val_loader)), end='\r')
            noise_batch = inferrer.get_noise(data)
            out_batch = inferrer.forward(data)

            # Plot noise
            plot_denoise(noise_ax_list, noise_batch)
            noise_fig_fname = 'figures/dae/mnist_dae_batch_%d_noise.png' % int(batch_idx)
            noise_fig.tight_layout()
            noise_fig.savefig(noise_fig_fname)

            # Plot outputs
            plot_denoise(out_ax_list, out_batch)
            out_fig_fname = 'figures/dae/mnist_dae_batch_%d_output.png' % int(batch_idx)
            out_fig.tight_layout()
            out_fig.savefig(out_fig_fname)

        print('\n OK')
        infer_end_time = time.time()
        infer_total_time = infer_end_time - infer_start_time
        print('Inferrer [%s] inferrer %d batches of size %d, total time : %s' %\
            (repr(inferrer), len(trainer.val_loader), trainer.batch_size, str(timedelta(seconds = infer_total_time)))
        )



    # Plot the loss history
    hist_fig, hist_ax = vis_loss_history.get_figure_subplots()
    vis_loss_history.plot_train_history_2subplots(
        hist_ax,
        trainer.get_loss_history(),
        cur_epoch = trainer.cur_epoch,
        iter_per_epoch = trainer.iter_per_epoch,
        loss_title = 'Denoising AE MNIST Training loss'
    )
    hist_fig.savefig(GLOBAL_OPTS['loss_history_file'], bbox_inches='tight')


def get_parser() -> argparse.ArgumentParser:
    parser = options.get_trainer_options()
    # General opts
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        default=False,
                        help='Set verbose mode'
                        )
    parser.add_argument('--infer',
                        action='store_true',
                        default=False,
                        help='Do an inference pass on the models after training'
                        )
    parser.add_argument('--data-dir',
                        type=str,
                        default='./data',
                        help='Path to location where data will be downloaded (default: ./data)'
                        )
    parser.add_argument('--loss-history-file',
                        type=str,
                        default='figures/dae_loss_history.png',
                        help='File to write loss history to (default: figures/dae_loss_history.png'
                        )
    # Noise options
    parser.add_argument('--noise-bias',
                        type=float,
                        default=0.25,
                        help='Amount to bias image by during noise application (default: 0.25)'
                        )
    parser.add_argument('--noise-factor',
                        type=float,
                        default=0.1,
                        help='Amount of noise to add to image overall (default: 0.1)'
                        )
    # Data options
    parser.add_argument('--checkpoint-dir',
                        type=str,
                        default='./checkpoint',
                        help='Set directory to place training snapshots into'
                        )
    parser.add_argument('--checkpoint-name',
                        type=str,
                        default='denoise_auto_mnist',
                        help='Name to prepend to all checkpoints'
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
