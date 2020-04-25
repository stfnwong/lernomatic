"""
TEST_FIND_LR
Unit test for learning rate finder

Stefan Wong 2019
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
# unit(s) under test
from lernomatic.param import lr_common
from lernomatic.train import cifar_trainer
from lernomatic.models import cifar
from lernomatic.vis import vis_loss_history
from test import util


# helper function for plotting
def plot_lr_find_results(ax,
                         loss_history:np.ndarray,
                         smooth_loss_history:np.ndarray,
                         lr_history:np.ndarray,
                         title:str='Learning rate finder output') -> None:
    ax.plot(np.arange(len(loss_history)), loss_history)
    ax.plot(np.arange(len(smooth_loss_history)), smooth_loss_history)
    ax.plot(np.arange(len(lr_history)), lr_history)
    ax.set_title(title)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss / Learning rate')
    ax.legend(['Loss', 'Smooth Loss', 'Learning rate'])


def plot_lr_vs_loss(ax, lr_history:np.ndarray, loss_history:np.ndarray) -> None:
    ax.plot(lr_history, loss_history)
    ax.set_title('Learning rate vs Loss')
    ax.set_xlabel('(Log) learning rate')
    ax.set_ylabel('(smoothed) loss')


def get_figure() -> tuple:
    fig, ax = plt.subplots()
    return (fig, ax)



GLOBAL_TEST_PARAMS = {
        'test_batch_size'        : 32,
        'test_learning_rate'     : 0.001,
        'test_lr_num_epochs'     : 4,            # number of epochs to run test for
        'test_print_every'       : 20,
        # options for learning rate finder
        'test_lr_min'            : 1e-8,
        'test_lr_max'            : 1.0,
        'test_num_iter'          : 5000,
        'test_lr_explode_thresh' : 4.5,
        'train_num_epochs'       : 80,
}


# Helper function to generate a trainer object
def get_trainer() -> cifar_trainer.CIFAR10Trainer:
    # get a model to test on and its corresponding trainer
    model = cifar.CIFAR10Net()
    trainer = cifar_trainer.CIFAR10Trainer(
        model,
        # turn off checkpointing
        save_every = 0,
        print_every = GLOBAL_TEST_PARAMS['test_print_every'],
        # data options
        batch_size = GLOBAL_TEST_PARAMS['test_batch_size'],
        # training options
        learning_rate = GLOBAL_TEST_PARAMS['test_learning_rate'],
        num_epochs = GLOBAL_TEST_PARAMS['train_num_epochs'],
        device_id = util.get_device_id(),
    )

    return trainer


class TestLogFinder:
    verbose          = True
    test_max_batches = 128
    draw_plot        = False

    def test_find_lr(self) -> None:
        # get an LRFinder
        trainer = get_trainer()
        lr_finder = lr_common.LogFinder(
            trainer,
            lr_min         = GLOBAL_TEST_PARAMS['test_lr_min'],
            lr_max         = GLOBAL_TEST_PARAMS['test_lr_max'],
            num_epochs     = GLOBAL_TEST_PARAMS['test_lr_num_epochs'],
            explode_thresh = GLOBAL_TEST_PARAMS['test_lr_explode_thresh'],
            max_batches    = self.test_max_batches,
            verbose        = self.verbose
        )

        if self.verbose:
            print('Created LRFinder object')
            print(lr_finder)

        lr_find_min, lr_find_max = lr_finder.find()
        print('Found learning rate range as %.3f -> %.3f' % (lr_find_min, lr_find_max))

        # show plot
        finder_fig, finder_ax = vis_loss_history.get_figure_subplots(2)
        lr_finder.plot_lr_vs_acc(finder_ax[0])
        lr_finder.plot_lr_vs_loss(finder_ax[1])
        if self.draw_plot is True:
            plt.show()
        else:
            plt.savefig('figures/test_find_lr_plots.png', bbox_inches='tight')

        # train the network with the discovered parameters
        trainer.print_every = 200
        trainer.train()

        train_fig, train_ax = vis_loss_history.get_figure_subplots()
        vis_loss_history.plot_train_history_2subplots(
            train_ax,
            trainer.get_loss_history(),
            acc_curve = trainer.get_acc_history(),
            iter_per_epoch = trainer.iter_per_epoch,
            cur_epoch = trainer.cur_epoch
        )
        if draw_plot is True:
            plt.show()
        else:
            train_fig.savefig('figures/test_find_lr_train_results.png', bbox_inches='tight')


    def test_lr_range_find(self) -> None:
        trainer = get_trainer()
        lr_finder = lr_common.LogFinder(
            trainer,
            lr_min     = GLOBAL_TEST_PARAMS['test_lr_min'],
            lr_max     = GLOBAL_TEST_PARAMS['test_lr_max'],
            num_iter   = GLOBAL_TEST_PARAMS['test_num_iter'],
            num_epochs = GLOBAL_TEST_PARAMS['test_lr_num_epochs'],
            acc_test   = True,
            max_batches    = self.test_max_batches,
            verbose    = self.verbose
        )

        # shut linter up
        if self.verbose:
            print(lr_finder)

        lr_find_min, lr_find_max = lr_finder.find()
        # show plot
        fig1, ax1 = plt.subplots()
        lr_finder.plot_lr_vs_acc(ax1)

        if draw_plot is True:
            plt.show()
        else:
            plt.savefig('figures/test_lr_range_find_lr_vs_acc.png', bbox_inches='tight')

        trainer.print_every = 200
        trainer.train()

        fig2, ax2 = vis_loss_history.get_figure_subplots()
        vis_loss_history.plot_train_history_2subplots(
            ax2,
            trainer.get_loss_history(),
            acc_curve = trainer.get_acc_history(),
            iter_per_epoch = trainer.iter_per_epoch,
            cur_epoch = trainer.cur_epoch
        )
        if draw_plot is True:
            plt.show()
        else:
            plt.savefig('figures/test_lr_range_find_train_results.png', bbox_inches='tight')
        # TODO : create some sort of data against which we can compare the test
        # outputs....


    def test_model_param_save(self) -> None:
        # get a trainer, etc
        trainer = get_trainer()
        lr_finder = lr_common.LogFinder(
            trainer,
            lr_min      = GLOBAL_TEST_PARAMS['test_lr_min'],
            lr_max      = GLOBAL_TEST_PARAMS['test_lr_max'],
            num_iter    = GLOBAL_TEST_PARAMS['test_num_iter'],
            num_epochs  = GLOBAL_TEST_PARAMS['test_lr_num_epochs'],
            acc_test    = True,
            max_batches = self.test_max_batches,
            verbose     = self.verbose
        )

        # shut linter up
        if self.verbose:
            print(lr_finder)

        # make a copy of the model parameters before we start looking for a new
        # learning rate.
        lr_find_min, lr_find_max = lr_finder.find()
        # show plot
        fig1, ax1 = plt.subplots()
        lr_finder.plot_lr_vs_acc(ax1)

        # now check that the restored parameters match the copy of the
        # parameters save earlier

        if draw_plot is True:
            plt.show()
        else:
            plt.savefig('figures/test_lr_range_find_lr_vs_acc.png', bbox_inches='tight')

        trainer.print_every = 200
        trainer.train()

        fig2, ax2 = vis_loss_history.get_figure_subplots()
        vis_loss_history.plot_train_history_2subplots(
            ax2,
            trainer.get_loss_history(),
            acc_curve = trainer.get_acc_history(),
            iter_per_epoch = trainer.iter_per_epoch,
            cur_epoch = trainer.cur_epoch
        )
        if draw_plot is True:
            plt.show()
        else:
            plt.savefig('figures/test_lr_range_find_train_results.png', bbox_inches='tight')


    def test_save_load(self) -> None:
        test_finder_state_file = 'data/test_lr_finder_state.pth'
        # get a trainer, etc
        trainer = get_trainer()
        src_lr_finder = lr_common.LogFinder(
            trainer,
            lr_min      = GLOBAL_TEST_PARAMS['test_lr_min'],
            lr_max      = GLOBAL_TEST_PARAMS['test_lr_max'],
            num_iter    = GLOBAL_TEST_PARAMS['test_num_iter'],
            num_epochs  = GLOBAL_TEST_PARAMS['test_lr_num_epochs'],
            acc_test    = True,
            max_batches = self.test_max_batches,
            verbose     = self.verbose
        )
        print('max_batches set to %d' % src_lr_finder.max_batches)

        # make a copy of the model parameters before we start looking for a new
        # learning rate.
        lr_find_min, lr_find_max = src_lr_finder.find()
        assert src_lr_finder.smooth_loss_history is not None
        # save the finder state and load into a new object
        src_lr_finder.save(test_finder_state_file)

        dst_lr_finder = lr_common.LogFinder(
            None,
            verbose    = self.verbose
        )
        dst_lr_finder.load(test_finder_state_file)
        # Since the trainer is not preserved in the save operation it makes no
        # sense to check it here

        # if this works, convert to dict and check
        assert src_lr_finder.lr_mult == dst_lr_finder.lr_mult
        assert src_lr_finder.lr_min == dst_lr_finder.lr_min
        assert src_lr_finder.lr_max == dst_lr_finder.lr_max
        assert src_lr_finder.explode_thresh == dst_lr_finder.explode_thresh
        assert src_lr_finder.beta == dst_lr_finder.beta
        assert src_lr_finder.gamma == dst_lr_finder.gamma
        assert src_lr_finder.lr_min_factor == dst_lr_finder.lr_min_factor
        assert src_lr_finder.lr_max_scale == dst_lr_finder.lr_max_scale
        assert src_lr_finder.lr_select_method == dst_lr_finder.lr_select_method

        # check histories
        print('Checking smooth loss history...', end=' ')
        assert len(src_lr_finder.smooth_loss_history) == len(dst_lr_finder.smooth_loss_history)
        for n in range(len(src_lr_finder.smooth_loss_history)):
             assert src_lr_finder.smooth_loss_history[n] == dst_lr_finder.smooth_loss_history[n]
        print(' OK')

        print('Checking log learning rate history...', end=' ')
        assert len(src_lr_finder.log_lr_history) == len(dst_lr_finder.log_lr_history)
        for n in range(len(src_lr_finder.log_lr_history)):
            assert src_lr_finder.log_lr_history[n] == dst_lr_finder.log_lr_history[n]
        print(' OK')

        print('Checking acc history...', end=' ')
        assert len(src_lr_finder.acc_history)== len(dst_lr_finder.acc_history)
        for n in range(len(src_lr_finder.acc_history)):
            assert src_lr_finder.acc_history[n] == dst_lr_finder.acc_history[n]
        print(' OK')
