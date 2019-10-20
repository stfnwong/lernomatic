"""
EX_AAE_MNIST
Adversairal Autoencoder example with MNIST dataset

Stefan Wong 2019
"""

import torch
import torchvision
import argparse
import numpy as np
import matplotlib.pyplot as plt
# tensorboard
from torch.utils import tensorboard
# timing stuff
import time
from datetime import timedelta

from lernomatic.infer.autoencoder import aae_inferrer
from lernomatic.train.autoencoder import aae_trainer
from lernomatic.train.autoencoder import aae_semisupervised_trainer
from lernomatic.models.autoencoder import aae_common
from lernomatic.models import common        # mainly for type hints
# command line options
from lernomatic.options import options

# MNIST subsample
from lernomatic.data.mnist import mnist_sub

# debug
#from pudb import set_trace; set_trace()


GLOBAL_OPTS = dict()
VALID_MODES = ('unsupervised', 'semisupervised')
MNIST_PARAMS = {
    'x_dim' : 784,
    'z_dim' : 2,
    'y_dim' : 10,
    'hidden_size' : 1024,
    'num_classes' : 10
}


def get_fig_subplots(num_subplots:int=2) -> tuple:
    fig = plt.figure()
    ax = []
    for p in range(num_subplots):
        sub_ax = fig.add_subplot(1, num_subplots, (p+1))
        ax.append(sub_ax)

    return (fig, ax)


def recon_to_plt(ax:list,
                 inp_img:torch.Tensor,
                 out_img:torch.Tensor,
                 img_dim:int=28,
                 ) -> None:
    if len(ax) < 2:
        raise ValueError('This method requires a list of at least 2 axis')

    ax[0].clear()
    ax[1].clear()

    # format arrays
    inp_arr = np.array(inp_img[0].detach().numpy()).reshape(img_dim, img_dim)
    out_arr = np.array(out_img[0].detach().numpy()).reshape(img_dim, img_dim)

    ax[0].imshow(inp_arr)
    ax[1].imshow(out_arr)
    ax[0].axis('off')
    ax[1].axis('off')
    ax[0].set_title('Input image')
    ax[1].set_title('Output image')


# TODO : once MNIST is working, move onto another dataset like celeba
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


def get_label_datasets(data_dir:str, k:int=3000) -> tuple:
    train_label_dataset = mnist_sub.MNISTSub(
        data_dir,
        train=True,
        download = True,
    )
    val_label_dataset = mnist_sub.MNISTSub(
        data_dir,
        train=False,
        download=True
    )

    return (train_label_dataset, val_label_dataset)


def get_semilabel_datasets(data_dir:str, k:int=3000, transform=mnist_sub.default_mnist_transform) -> tuple:
    return mnist_sub.gen_mnist_subset(
        data_dir,
        transform=transform,
        verbose = GLOBAL_OPTS['verbose']
    )


def infer_aae(
    q_net:common.LernomaticModel,
    p_net:common.LernomaticModel,
    batch_size:int,
    val_dataset,
    aae_mode:str='supervised') -> None:

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size,
        drop_last = True,
        shuffle = False,
    )

    # Now run inference and generate some examples
    if aae_mode == 'unsupervised':
        inferrer = aae_inferrer.AAEUnsupervisedInferrer(
            q_net,
            p_net,
            device_id = GLOBAL_OPTS['device_id']
        )
    elif aae_mode == 'semisupervised':
        inferrer = aae_inferrer.AAESemiInferrer(
            q_net,
            p_net,
            device_id = GLOBAL_OPTS['device_id']
        )
    else:
        raise NotImplementedError('Mode [%s] not implemented' % str(aae_mode))

    # grab some data from the val_loader and push through inferrer
    fig, ax = get_fig_subplots(num_subplots = 2)

    for batch_idx, (data, target) in enumerate(val_loader):
        print('Processing validation example [%d / %d]' % (batch_idx+1, len(val_loader)), end='\r')
        #data.resize_(batch_size, inferrer.q_net.get_x_dim())
        gen_img = inferrer.forward(data)
        img_filename = 'figures/aae_%s_batch_%d.png' % (str(aae_mode), int(batch_idx))
        recon_to_plt(ax, data.cpu(), gen_img.cpu())
        fig.savefig(img_filename, bbox_inches='tight')

    print('\n OK')


def unsupervised() -> None:
    # get some models
    q_net = aae_common.AAEQNet(MNIST_PARAMS['x_dim'], MNIST_PARAMS['z_dim'], MNIST_PARAMS['hidden_size'])
    p_net = aae_common.AAEPNet(MNIST_PARAMS['x_dim'], MNIST_PARAMS['z_dim'], MNIST_PARAMS['hidden_size'])
    d_net = aae_common.AAEDNetGauss(MNIST_PARAMS['z_dim'], MNIST_PARAMS['hidden_size'])

    # get some data
    train_dataset, val_dataset = get_datasets(GLOBAL_OPTS['data_dir'])

    # get a trainer
    trainer = aae_trainer.AAETrainer(
        q_net,
        p_net,
        d_net,
        # datasets
        train_dataset = train_dataset,
        val_dataset   = val_dataset,
        # train options
        num_epochs    = GLOBAL_OPTS['num_epochs'],
        batch_size    = GLOBAL_OPTS['batch_size'],
        # misc
        print_every   = GLOBAL_OPTS['print_every'],
        save_every    = GLOBAL_OPTS['save_every'],
        device_id     = GLOBAL_OPTS['device_id'],
        verbose       = GLOBAL_OPTS['verbose']
    )
    # Optionally get a summary writer
    if GLOBAL_OPTS['tensorboard_dir'] is not None:
        writer = tensorboard.SummaryWriter()
        trainer.set_tb_writer(writer)

    # We would do parameter search and scheduling here, but I will leave these
    # for now since I need to read more about how to train effectively with
    # GANs
    train_start_time = time.time()
    trainer.train()
    train_end_time = time.time()
    train_total_time = train_end_time - train_start_time
    print('Trained [%s] for %d epochs, total time : %s' %\
          (repr(trainer), trainer.cur_epoch, str(timedelta(seconds = train_total_time)))
    )

    infer_aae(
        q_net,
        p_net,
        GLOBAL_OPTS['batch_size'],
        val_dataset,
        aae_mode = GLOBAL_OPTS['mode']
    )


def semisupervised() -> None:
    q_net = aae_common.AAEQNet(
        MNIST_PARAMS['x_dim'],
        MNIST_PARAMS['z_dim'],
        MNIST_PARAMS['hidden_size'],
        num_classes = MNIST_PARAMS['num_classes']
    )
    p_net = aae_common.AAEPNet(
        MNIST_PARAMS['x_dim'],
        MNIST_PARAMS['z_dim'] + MNIST_PARAMS['num_classes'],
        MNIST_PARAMS['hidden_size']
    )
    d_cat_net = aae_common.AAEDNetGauss(
        MNIST_PARAMS['num_classes'],
        MNIST_PARAMS['hidden_size']
    )
    d_gauss_net = aae_common.AAEDNetGauss(
        MNIST_PARAMS['z_dim'],
        MNIST_PARAMS['hidden_size']
    )

    # We also need to sub-sample some parts of the MNIST dataset to produce the
    # 'labelled' data loaders
    train_label_dataset, val_label_dataset, train_unlabel_dataset = \
        get_semilabel_datasets(GLOBAL_OPTS['data_dir'])

    # get a trainer
    trainer = aae_semisupervised_trainer.AAESemiTrainer(
        q_net,
        p_net,
        d_cat_net,
        d_gauss_net,
        # datasets
        train_label_dataset   = train_label_dataset,
        train_unlabel_dataset = train_unlabel_dataset,
        val_label_dataset     = val_label_dataset,
        # train options
        num_epochs    = GLOBAL_OPTS['num_epochs'],
        batch_size    = GLOBAL_OPTS['batch_size'],
        # misc
        print_every   = GLOBAL_OPTS['print_every'],
        save_every    = GLOBAL_OPTS['save_every'],
        device_id     = GLOBAL_OPTS['device_id'],
        verbose       = GLOBAL_OPTS['verbose']
    )

    # Optionally get a summary writer
    if GLOBAL_OPTS['tensorboard_dir'] is not None:
        writer = tensorboard.SummaryWriter()
        trainer.set_tb_writer(writer)

    train_start_time = time.time()
    trainer.train()
    train_end_time = time.time()
    train_total_time = train_end_time - train_start_time
    print('Trained [%s] for %d epochs, total time : %s' %\
          (repr(trainer), trainer.cur_epoch, str(timedelta(seconds = train_total_time)))
    )

    # now do an inference pass...
    infer_aae(
        q_net,
        p_net,
        GLOBAL_OPTS['batch_size'],
        val_label_dataset,
        aae_mode = GLOBAL_OPTS['mode']
    )

def supervised() -> None:
    raise NotImplementedError('This mode not yet implemented')


def get_parser() -> argparse.ArgumentParser:
    parser = options.get_trainer_options()
    # General opts
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        default=False,
                        help='Set verbose mode'
                        )
    parser.add_argument('--mode',
                        type=str,
                        default='unsupervised',
                        help='Tool mode. Must be one of %s (default: unsupervised)' % str(VALID_MODES)
                        )
    parser.add_argument('--data-dir',
                        type=str,
                        default='./data',
                        help='Path to location where data will be downloaded (default: ./data)'
                        )
    # Data options
    parser.add_argument('--checkpoint-dir',
                        type=str,
                        default='./checkpoint',
                        help='Set directory to place training snapshots into'
                        )
    parser.add_argument('--checkpoint-name',
                        type=str,
                        default='aae',
                        help='Name to prepend to all checkpoints'
                        )
    parser.add_argument('--tensorboard-dir',
                        default=None,
                        type=str,
                        help='Directory to save tensorboard runs to. If None, tensorboard is not used. (default: None)'
                        )



    return parser



if __name__ == '__main__':
    parser = get_parser()
    opts   = vars(parser.parse_args())

    for k, v in opts.items():
        GLOBAL_OPTS[k] = v

    GLOBAL_OPTS['checkpoint_name'] =\
        str(GLOBAL_OPTS['checkpoint_name']) + '_' + GLOBAL_OPTS['mode']

    if GLOBAL_OPTS['verbose'] is True:
        print(' ---- GLOBAL OPTIONS ---- ')
        for k,v in GLOBAL_OPTS.items():
            print('\t[%s] : %s' % (str(k), str(v)))

    if GLOBAL_OPTS['mode'] == 'unsupervised':
        unsupervised()
    elif GLOBAL_OPTS['mode'] == 'semisupervised':
        semisupervised()
    elif GLOBAL_OPTS['mode'] == 'supervised':
        supervised()
    else:
        raise ValueError('Unsupported tool mode [%s]' % str(GLOBAL_OPTS['mode']))
