"""
EX_LR_SCHEDULE_CIFAR10
CIFAR-10 example using the LR Scheduler

Stefan Wong 2019
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from lernomatic.param import learning_rate
from lernomatic.vis import vis_lr
# we use CIFAR-10 for this example
from lernomatic.models import cifar10
from lernomatic.train import cifar10_trainer
from lernomatic.train import schedule
# vis tools
from lernomatic.vis import vis_loss_history

# debug
#from pudb import set_trace; set_trace()

GLOBAL_OPTS = dict()


def differentiate(function):
    dx = np.zeros(len(function))
    for n in range(len(function)-1):
        dx[n] = function[n+1] - function[n]

    return dx

#self.avg_loss = self.beta * self.avg_loss + (1.0 - self.beta) * loss.item()
#smooth_loss = self.avg_loss / (1.0 - self.beta ** (batch_idx+1))
def differentiate_smooth(function, beta=0.98):
    dx_avg = 0.0
    dx = np.zeros(len(function))
    for n in range(len(function)-1):
        dx_avg = beta * dx_avg + (1.0 + beta) * function[n]
        dx[n] = function[n+1] - function[n]

    return dx


def main():

    # get a model and trainer
    model = cifar10.CIFAR10Net()
    #model = resnets.WideResnet(28, 10)
    trainer = cifar10_trainer.CIFAR10Trainer(
        model,
        batch_size      = GLOBAL_OPTS['batch_size'],
        test_batch_size = GLOBAL_OPTS['test_batch_size'],
        num_epochs      = GLOBAL_OPTS['num_epochs'],
        learning_rate   = GLOBAL_OPTS['learning_rate'],
        #momentum = GLOBAL_OPTS['momentum'],
        weight_decay    = GLOBAL_OPTS['weight_decay'],
        # device
        device_id       = GLOBAL_OPTS['device_id'],
        # checkpoint
        checkpoint_dir  = GLOBAL_OPTS['checkpoint_dir'],
        checkpoint_name = 'ex_cifar10_lr_find_schedule_',
        # display,
        print_every     = GLOBAL_OPTS['print_every'],
        save_every      = GLOBAL_OPTS['save_every'],
        verbose         = GLOBAL_OPTS['verbose']
    )

    # get an LRFinder object
    lr_finder = learning_rate.LogFinder(
        trainer,
        lr_min         = GLOBAL_OPTS['lr_min'],
        lr_max         = GLOBAL_OPTS['lr_max'],
        num_epochs     = GLOBAL_OPTS['find_num_epochs'],
        explode_thresh = GLOBAL_OPTS['find_explode_thresh'],
        print_every    = GLOBAL_OPTS['find_print_every']
    )
    print(lr_finder)

    lr_finder.find()

    dx_smooth_loss = differentiate(lr_finder.loss_grad_history)
    loss_fig, loss_ax = plt.subplots()

    loss_ax.plot(np.arange(len(lr_finder.smooth_loss_history)), lr_finder.smooth_loss_history, 'b')
    loss_ax.set_xlabel('Iteration')
    loss_ax.set_ylabel('Loss')
    loss_ax.set_title('LR finder Smoothed loss')
    loss_fig.savefig('figures/ex_lr_finder_loss.png', bbox_inches='tight')

    dx_fig, dx_ax = plt.subplots()
    dx_ax.plot(np.arange(len(lr_finder.loss_grad_history)), lr_finder.loss_grad_history, 'g')
    dx_ax.plot(np.arange(len(dx_smooth_loss)), dx_smooth_loss, 'r')
    dx_ax.set_xlabel('Iteration')
    dx_ax.set_ylabel('Smoothed loss')
    dx_ax.set_title('LR Finder Derivative of smoothed loss')
    dx_ax.legend(['loss dx', 'loss d2x'])
    dx_fig.savefig('figures/ex_lr_finder_loss_dx.png', bbox_inches='tight')

    # LR vs loss
    lr_loss_fig, lr_loss_ax = plt.subplots()
    lr_loss_ax.plot(lr_finder.log_lr_history, lr_finder.smooth_loss_history)
    lr_loss_ax.set_xlabel('Learning rate (log)')
    lr_loss_ax.set_ylabel('Smooth loss')
    lr_loss_ax.set_title('(Log) Learning rate vs. Smooth Loss')
    lr_loss_fig.savefig('figures/ex_lr_finder_lr_loss.png', bbox_inches='tight')

    # also create a graph of LR vs acc
    if lr_finder.acc_test is True:
        lr_fig, lr_ax = plt.subplots()
        vis_lr.plot_lr_vs_acc(
            lr_ax,
            lr_finder.log_lr_history,
            lr_finder.acc_history
        )
        lr_fig.savefig('figures/ex_lr_finder_lr_vs_acc.png', bbox_inches='tight')

    # try to find the critical points on the graph
    # TODO : automatic setting!
    lr_find_max = 1e-1
    lr_find_min = 1e-2

    lr_scheduler = schedule.TriangularScheduler(
        stepsize = int(len(trainer.train_loader) / 4),
        lr_min = lr_find_min,
        lr_max = lr_find_max
    )
    assert(trainer.acc_iter == 0)

    print('Adding scheduler to trainer')
    trainer.set_lr_scheduler(lr_scheduler)
    trainer.train()

    # show the training results
    train_fig, train_ax = vis_loss_history.get_figure_subplots()
    vis_loss_history.plot_train_history_2subplots(
        train_ax,
        trainer.loss_history,
        acc_history = trainer.acc_history[0 : trainer.acc_iter],
        cur_epoch = trainer.cur_epoch,
        iter_per_epoch = trainer.iter_per_epoch,
        loss_title = 'CIFAR-10 LR Finder Loss\n (%s min LR: %f, max LR: %f' % (repr(lr_scheduler), lr_scheduler.lr_min, lr_scheduler.lr_max),
        acc_title = 'CIFAR-10 LR Finder Accuracy '
    )
    train_fig.savefig('figures/ex_lr_finder_train_output.png', bbox_inches='tight')

    # How does this compare to training without the scheduler?
    print('Creating trainer with no scheduler...')
    no_sched_model = cifar10.CIFAR10Net()
    no_sched_trainer = cifar10_trainer.CIFAR10Trainer(
        no_sched_model,
        batch_size      = GLOBAL_OPTS['batch_size'],
        test_batch_size = GLOBAL_OPTS['test_batch_size'],
        num_epochs      = GLOBAL_OPTS['num_epochs'],
        learning_rate   = 0.001,
        #momentum = GLOBAL_OPTS['momentum'],
        weight_decay    = GLOBAL_OPTS['weight_decay'],
        # device
        device_id       = GLOBAL_OPTS['device_id'],
        # checkpoint
        checkpoint_dir  = GLOBAL_OPTS['checkpoint_dir'],
        checkpoint_name = 'ex_cifar10_lr_find_no_schedule_',
        # display,
        print_every     = GLOBAL_OPTS['print_every'],
        save_every      = GLOBAL_OPTS['save_every'],
        verbose         = GLOBAL_OPTS['verbose']
    )
    assert(no_sched_trainer.acc_iter == 0)

    no_sched_trainer.train()

    # show the training results without scheduling
    ns_train_fig, ns_train_ax = vis_loss_history.get_figure_subplots()
    vis_loss_history.plot_train_history_2subplots(
        ns_train_ax,
        no_sched_trainer.loss_history,
        acc_history = no_sched_trainer.acc_history[0 : no_sched_trainer.acc_iter],
        cur_epoch = no_sched_trainer.cur_epoch,
        iter_per_epoch = no_sched_trainer.iter_per_epoch,
        loss_title = 'CIFAR-10 LR Finder Loss (no scheduling, lr=%f' % no_sched_trainer.get_learning_rate(),
        acc_title = 'CIFAR-10 LR Finder Accuracy  (no scheduling)'
    )
    ns_train_fig.savefig('figures/ex_lr_finder_train_no_sched_output.png', bbox_inches='tight')


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
    parser.add_argument('--find-print-every',
                        type=int,
                        default=20,
                        help='How often to print output from learning rate finder'
                        )
    parser.add_argument('--find-num-epochs',
                        type=int,
                        default=8,
                        help='Maximum number of epochs to attempt to find learning rate'
                        )
    parser.add_argument('--find-explode-thresh',
                        type=float,
                        default=4.5,
                        help='Threshold at which to stop increasing learning rate'
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
    parser.add_argument('--batch-size',
                        type=int,
                        default=64,
                        help='Batch size to use during training'
                        )
    parser.add_argument('--test-batch-size',
                        type=int,
                        default=64,
                        help='Batch size to use during testing'
                        )
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

    parser.add_argument('--weight-decay',
                        type=float,
                        default=0.0,
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
