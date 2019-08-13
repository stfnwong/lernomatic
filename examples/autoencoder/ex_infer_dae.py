"""
EX_INFER_DAE
Perform inference on Denoising Autoencoder data

Stefan Wong 2019
"""

import argparse
import torch
import torchvision
import matplotlib.pyplot as plt

# inferrers, etc
from lernomatic.infer.autoencoder import dae_inferrer

# vis options
from lernomatic.vis import vis_loss_history
from lernomatic.vis import vis_img
from lernomatic.util import image_util


GLOBAL_OPTS = dict()

from pudb import set_trace; set_trace()


def plot_denoise(
    ax,
    image_batch:torch.Tensor) -> None:

    if len(ax) != image_batch.shape[0]:
        raise ValueError('Number of image axes (%d) does not equal batch size (%d)' %\
                         (len(ax), image_batch.shape[0])
        )

    for img_idx in range(image_batch.shape[0]):
        img = image_util.tensor_to_img(image_batch[img_idx, :, :, :])
        if img.shape[-1] == 1:
            img = img.squeeze(len(img.shape)-1)
        ax[img_idx].imshow(img)
        ax[img_idx].get_xaxis().set_visible(False)
        ax[img_idx].get_yaxis().set_visible(False)


def main() -> None:
    # TODO: just using MNIST for now, but need to generalize for other data
    dataset_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize( (0.1307,), (0.3081,))
    ])
    val_dataset = torchvision.datasets.MNIST(
        GLOBAL_OPTS['data_dir'],
        train = False,
        download = True,
        transform = dataset_transform
    )

    # Get an inferrer
    inferrer = dae_inferrer.DAEInferrer(
        None,
        None,
        noise_bias   = GLOBAL_OPTS['noise_bias'],
        noise_factor = GLOBAL_OPTS['noise_factor'],
        device_id    = GLOBAL_OPTS['device_id'],
        verbose      = GLOBAL_OPTS['verbose']
    )
    inferrer.load_model(GLOBAL_OPTS['checkpoint'])

    if inferrer.encoder is None:
        raise ValueError('Failed to load encoder from file [%s]' % str(GLOBAL_OPTS['checkpoint']))
    if inferrer.decoder is None:
        raise ValueError('Failed to load decoder from file [%s]' % str(GLOBAL_OPTS['checkpoint']))

    # Create a loader for the dataset
    batch_size = GLOBAL_OPTS['subplot_size'] * GLOBAL_OPTS['subplot_size']
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size = batch_size,
        drop_last  = True,
        shuffle    = False
    )

    # get figures, axes
    out_fig, out_ax_list = vis_img.get_grid_subplots(GLOBAL_OPTS['subplot_size'])
    noise_fig, noise_ax_list = vis_img.get_grid_subplots(GLOBAL_OPTS['subplot_size'])

    for batch_idx, (data, _) in enumerate(val_loader):
        print('Inferring batch [%d / %d]' % (batch_idx+1, len(val_loader)), end='\r')
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



def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    # Checkpoint options
    parser.add_argument('checkpoint',
                        type=str,
                        default='denoise_auto_',
                        help='Name to prepend to all checkpoints'
                        )
    # General opts
    parser.add_argument('--data-dir',
                        type=str,
                        default='./data/',
                        help='Path to directory to store data (default: ./data/)'
                        )
    parser.add_argument('--device-id',
                        type=int,
                        default=-1,
                        help='Device ID for device to infer on (default: -1)'
                        )
    parser.add_argument('--subplot-size',
                        type=int,
                        default=8,
                        help='Size of output image square to use (default: 8)'
                        )
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        default=False,
                        help='Set verbose mode'
                        )
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
