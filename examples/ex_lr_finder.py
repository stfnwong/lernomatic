"""
EX_LR_FINDER
Example use of the learning rate finder

Stefan Wong 2019
"""

import argparse
import matplotlib.pyplot as plt
from lernomatic.param import learning_rate
from lernomatic.vis import vis_lr
# we use CIFAR-10 for this example
from lernomatic.models import cifar10
from lernomatic.models import resnets
from lernomatic.train import cifar10_trainer

from lernomatic.vis import vis_loss_history

GLOBAL_OPTS = dict()


def main():

    # get a model and trainer
    #model = cifar10.CIFAR10Net()
    model = resnets.WideResnet(28, 10)
    trainer = cifar10_trainer.CIFAR10Trainer(
        model,
        batch_size = GLOBAL_OPTS['batch_size'],
        num_epochs = GLOBAL_OPTS['num_epochs'],
        learning_rate = GLOBAL_OPTS['learning_rate'],
        #momentum = GLOBAL_OPTS['momentum'],
        #weight_decay = GLOBAL_OPTS['weight_decay'],
        # device
        device_id = GLOBAL_OPTS['device_id'],
        # checkpoint
        checkpoint_dir = GLOBAL_OPTS['checkpoint_dir'],
        checkpoint_name = GLOBAL_OPTS['checkpoint_name'],
        # display,
        print_every = 2,
        save_every = GLOBAL_OPTS['save_every'],
        verbose = GLOBAL_OPTS['verbose']
    )

    # get an LRFinder object
    lr_finder = learning_rate.LinearFinder(
        len(trainer.train_loader),
        lr_min       = GLOBAL_OPTS['lr_min'],
        lr_max       = GLOBAL_OPTS['lr_max'],
        num_epochs = GLOBAL_OPTS['max_epochs'],
    )
    trainer.print_every = GLOBAL_OPTS['print_every']
    trainer.find_lr(lr_finder)
    trainer.train()

    if GLOBAL_OPTS['verbose']:
        print('Created new %s object ' % repr(lr_finder))
        print(lr_finder)


    fig1, ax1 = plt.subplots()
    vis_lr.plot_lr_vs_acc(ax1, None, None, title='Learning rate finder example')
    if GLOBAL_OPTS['draw_plot']:
        plt.show()
    else:
        plt.savefig('lr_finder_lr_vs_acc.png', bbox_inches='tight')

    fig2, ax2 = plt.subplots()
    vis_loss_history.plot_loss_history(
        ax2,
        trainer.loss_history,
        acc_curve = trainer.acc_history,
        iter_per_epoch = trainer.iter_per_epoch,
        cur_epoch = trainer.cur_epoch
    )
    if GLOBAL_OPTS['draw_plot']:
        plt.show()
    else:
        plt.savefig('lr_finder_train_results.png', bbox_inches='tight')

def get_parser():
    parser = argparse.ArgumentParser()
    # General opts
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        default=False,
                        help='Set verbose mode'
                        )
    parser.add_argument('--draw-plot',
                        default=False,
                        action='store_true',
                        help='Display plots'
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
    # Learning rate finder options
    parser.add_argument('--max-epochs',
                        type=int,
                        default=100,
                        help='Maximum number of epochs to attempt to find learning rate'
                        )
    parser.add_argument('--lr-min',
                        type=float,
                        default=2e-4,
                        help='Minimum range to search for learning rate'
                        )
    parser.add_argument('--lr-max',
                        type=float,
                        default=1e-1,
                        help='Maximum range to search for learning rate'
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
    # Data options
    parser.add_argument('--checkpoint-dir',
                        type=str,
                        default='./checkpoint',
                        help='Set directory to place training snapshots into'
                        )
    parser.add_argument('--checkpoint-name',
                        type=str,
                        default='lr_find_ex_cifar10',
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
