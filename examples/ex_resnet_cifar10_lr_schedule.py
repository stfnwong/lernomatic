"""
EX_RESET_CIFAR10_LR_SCHEDULE
Example using LRScheduler objects with a resnet trained
on CIFAR10

Stefan Wong 2019
"""
import sys
import argparse

from lernomatic.train import schedule
from lernomatic.train import resnet_trainer
from lernomatic.models import resnet
from lernomatic.param import learning_rate

# debug
from pudb import set_trace; set_trace()

GLOBAL_OPTS = dict()

def main():

    if GLOBAL_OPTS['load_checkpoint'] is not None:
        print('Loading checkpoints is not yet implemented')

    # get a model
    model = resnet.WideResnet(28, 10)
    trainer = resnet_trainer.ResnetTrainer(
        model,

        # training time
        num_epochs = GLOBAL_OPTS['num_epochs'],

        # other
        device_id = GLOBAL_OPTS['device_id'],
        verbose = GLOBAL_OPTS['verbose']

    )

    # prepare lr_finder
    lr_finder = learning_rate.LinearFinder(
        len(trainer.train_loader),
        lr_min = GLOBAL_OPTS['lr_min'],
        lr_max = GLOBAL_OPTS['lr_max']
    )

    # prepare learning schedule
    lr_schedule = schedule.TriangularSchedule(
        stepsize = int(len(trainer.train_loader) / 2)
    )

    trainer.set_schedule(lr_schedule)



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
                        default=2e-8,
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
