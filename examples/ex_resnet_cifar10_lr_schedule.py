"""
EX_RESET_CIFAR10_LR_SCHEDULE
Example using LRScheduler objects with a resnet trained
on CIFAR10

Stefan Wong 2019
"""
import sys
import argparse
import matplotlib.pyplot as plt

from lernomatic.train import schedule
from lernomatic.train import resnet_trainer
from lernomatic.models import resnets
from lernomatic.param import learning_rate
# vis stuff
from lernomatic.vis import vis_loss_history

# debug
#from pudb import set_trace; set_trace()

GLOBAL_OPTS = dict()


def main():
    if GLOBAL_OPTS['load_checkpoint'] is not None:
        print('Loading checkpoints is not yet implemented')

    # get a model
    model = resnets.WideResnet(GLOBAL_OPTS['resnet_depth'], 10)
    trainer = resnet_trainer.ResnetTrainer(
        model,
        # training time
        num_epochs      = GLOBAL_OPTS['num_epochs'],
        learning_rate   = GLOBAL_OPTS['learning_rate'],   # this will be overwritten by lr_finder later
        batch_size      = GLOBAL_OPTS['batch_size'],
        # other
        device_id       = GLOBAL_OPTS['device_id'],
        verbose         = GLOBAL_OPTS['verbose'],
        # checkpoint options
        save_every      = GLOBAL_OPTS['save_every'],
        checkpoint_name = GLOBAL_OPTS['checkpoint_name'],
        checkpoint_dir  = GLOBAL_OPTS['checkpoint_dir'],
    )

    # prepare lr_finder
    r_finder = learning_rate.LogFinder(
        trainer,
        lr_min         = GLOBAL_OPTS['lr_min'],
        lr_max         = GLOBAL_OPTS['lr_max'],
        num_epochs     = GLOBAL_OPTS['find_num_epochs'],
        explode_thresh = GLOBAL_OPTS['find_explode_thresh'],
        print_every    = 20
    )
    lr_finder.find()

    # prepare learning schedule
    lr_schedule = schedule.TriangularScheduler(
        stepsize = int(len(trainer.train_loader) / 4)
    )

    trainer.set_lr_scheduler(lr_schedule)
    trainer.train()

    # Plot outputs (optional)
    train_fig, train_ax = vis_loss_history.get_figure_subplots()
    vis_loss_history.plot_train_history_2subplots(
        train_ax,
        trainer.loss_history,
        acc_history = trainer.acc_history,
        cur_epoch = trainer.cur_epoch,
        iter_per_epoch = trainer.iter_per_epoch,
        loss_title = 'CIFAR-10 Resnet LR Finder Loss (example with scheduling)',
        acc_title = 'CIFAR-10 Resnet LR Finder Accuracy (example with scheduling)'
    )
    train_fig.savefig('figures/resnet_cifar10_lr_schedule.png')

    if GLOBAL_OPTS['draw_plot']:
        plt.show()


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
    parser.add_argument('--find-num-epochs',
                        type=int,
                        default=10,
                        help='Maximum number of epochs to attempt to find learning rate'
                        )
    parser.add_argument('--find-explode-thresh',
                        type=float,
                        default=4.0,
                        help='Maximum number of epochs to attempt to find learning rate'
                        )
    parser.add_argument('--lr-min',
                        type=float,
                        default=2e-8,
                        help='Minimum range to search for learning rate'
                        )
    parser.add_argument('--lr-max',
                        type=float,
                        default=5e-1,
                        help='Maximum range to search for learning rate'
                        )
    # Device options
    parser.add_argument('--device-id',
                        type=int,
                        default=-1,
                        help='Set device id (-1 for CPU)'
                        )
    # Network options
    parser.add_argument('--resnet-depth',
                         type=int,
                         default=28,
                         help='Number of layers to use for Resnet'
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
    parser.add_argument('--checkpoint-dir',
                        type=str,
                        default='./checkpoint',
                        help='Set directory to place training snapshots into'
                        )
    parser.add_argument('--checkpoint-name',
                        type=str,
                        default='resnet_cifar10_schedule',
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
