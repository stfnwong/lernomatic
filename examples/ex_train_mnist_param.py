"""
EX_TRAIN_MNIST
Train a classifier on the MNIST handwritten digits example

Stefan Wong 2019
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from lernomatic.train import mnist_trainer
from lernomatic.models import mnist
# add learning rate scheduler
from lernomatic.train import schedule
from lernomatic.param import lr_common

# debug
#from pudb import set_trace; set_trace()

GLOBAL_OPTS = dict()

def differentiate(function):
    dx = np.zeros(len(function))
    for n in range(len(function)-1):
        dx[n] = function[n+1] - function[n]

    return dx

def main():

    # Get a model
    model = mnist.MNISTNet()

    # Get a trainer
    trainer = mnist_trainer.MNISTTrainer(
        model,
        # training parameters
        batch_size = GLOBAL_OPTS['batch_size'],
        num_epochs = GLOBAL_OPTS['num_epochs'],
        learning_rate = GLOBAL_OPTS['learning_rate'],
        momentum = GLOBAL_OPTS['momentum'],
        weight_decay = GLOBAL_OPTS['weight_decay'],
        # device
        device_id = GLOBAL_OPTS['device_id'],
        # checkpoint
        checkpoint_dir = GLOBAL_OPTS['checkpoint_dir'],
        checkpoint_name = GLOBAL_OPTS['checkpoint_name'],
        # display,
        print_every = GLOBAL_OPTS['print_every'],
        save_every = GLOBAL_OPTS['save_every'],
        verbose = GLOBAL_OPTS['verbose']
    )

    # get an LRFinder object
    lr_finder = lr_common.LogFinder(
        trainer,
        lr_min         = 1e-8,
        lr_max         = 1.0,
        num_epochs     = 3,
        explode_thresh = 10,
        print_every    = 20
    )
    print(lr_finder)

    lr_finder.find()
    lr_min, lr_max = lr_finder.get_lr_range()
    print('Found LR range as %.4f -> %.4f' % (lr_min, lr_max))

    dx_smooth_loss = differentiate(lr_finder.loss_grad_history)
    loss_fig, loss_ax = plt.subplots()
    loss_ax.plot(np.arange(len(lr_finder.smooth_loss_history)), lr_finder.smooth_loss_history, 'b')
    loss_ax.plot(np.arange(len(lr_finder.loss_grad_history)), lr_finder.loss_grad_history, 'g')
    loss_ax.plot(np.arange(len(dx_smooth_loss)), dx_smooth_loss, 'r')
    loss_ax.set_xlabel('Iteration')
    loss_ax.set_ylabel('2nd order smoothed loss')
    loss_ax.set_title('Second derivative of smoothed loss')
    loss_ax.legend(['smoothed loss', 'smooth loss dx', 'smooth loss d2x'])
    loss_fig.savefig('figures/ex_lr_finder_smooth_loss_d2x.png', bbox_inches='tight')

    #lr_schedule = schedule.TriangularLRSchedule(
    #)
    trainer.train()


def get_parser():
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
    parser.add_argument('--momentum',
                        type=float,
                        default=0.5,
                        help='Momentum for SGD'
                        )
    parser.add_argument('--grad-clip',
                        type=float,
                        default=5.0,
                        help='Clip gradients at this (absolute) value'
                        )
    # Data options
    parser.add_argument('--checkpoint-dir',
                        type=str,
                        default='./checkpoint',
                        help='Set directory to place training snapshots into'
                        )
    parser.add_argument('--checkpoint-name',
                        type=str,
                        default='mnist',
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
            print('%s : %s' % (str(k), str(v)))

    main()
