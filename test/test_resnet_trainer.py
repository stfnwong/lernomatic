"""
TEST_RESNET_TRAINER
Test the resnet trainer object.

Stefan Wong 2019
"""

import torch
import matplotlib.pyplot as plt
# units under test
from lernomatic.train import resnet_trainer
from lernomatic.models import resnets
from lernomatic.vis import vis_loss_history
from test import util


GLOBAL_OPTS = dict()


class TestResnetTrainer:
    verbose = True
    resnet_depth = 28
    test_batch_size = 64
    test_num_epochs = 2
    test_learning_rate = 0.001
    checkpoint_dir = 'checkpoint/'
    print_every = 100

    def test_save_load_checkpoint(self) -> None:
        test_checkpoint_name = self.checkpoint_dir + 'resnet_trainer_checkpoint.pkl'
        test_history_name    = self.checkpoint_dir + 'resnet_trainer_history.pkl'
        # get a model
        model = resnets.WideResnet(
            depth = self.resnet_depth,
            num_classes = 10,     # using CIFAR-10 data
            input_channels=3,
            w_factor = 1
        )
        # get a traner
        src_tr = resnet_trainer.ResnetTrainer(
            model,
            # training parameters
            batch_size    = self.test_batch_size,
            num_epochs    = self.test_num_epochs,
            learning_rate = self.test_learning_rate,
            # device
            device_id     = util.get_device_id(),
            # display,
            print_every   = self.print_every,
            save_every    = 0,
            verbose       = self.verbose
        )

        if self.verbose:
            print('Created %s object' % repr(src_tr))
            print(src_tr)

        print('Training model %s for %d epochs' % (repr(src_tr), self.test_num_epochs))
        src_tr.train()

        # save the final checkpoint
        src_tr.save_checkpoint(test_checkpoint_name)
        src_tr.save_history(test_history_name)

        # get a new trainer and load checkpoint
        dst_tr = resnet_trainer.ResnetTrainer(
            model
        )
        dst_tr.load_checkpoint(test_checkpoint_name)

        # Test object parameters
        assert src_tr.num_epochs == dst_tr.num_epochs
        assert src_tr.learning_rate == dst_tr.learning_rate
        assert src_tr.weight_decay == dst_tr.weight_decay
        assert src_tr.print_every == dst_tr.print_every
        assert src_tr.save_every == dst_tr.save_every

        print('\t Comparing model parameters ')
        src_model_params = src_tr.get_model_params()
        dst_model_params = dst_tr.get_model_params()
        assert len(src_model_params.items()) == len(dst_model_params.items())

        # p1, p2 are k,v tuple pairs of each model parameters
        # k = str
        # v = torch.Tensor
        for n, (p1, p2) in enumerate(zip(src_model_params.items(), dst_model_params.items())):
            assert p1[0] == p2[0]
            print('Checking parameter %s [%d/%d] \t\t' % (str(p1[0]), n+1, len(src_model_params.items())), end='\r')
            assert torch.equal(p1[1], p2[1]) == True
        print('\n ...done')

        # test history
        dst_tr.load_history(test_history_name)

        # loss history
        assert len(src_tr.loss_history) == len(dst_tr.loss_history)
        assert src_tr.loss_iter == dst_tr.loss_iter
        for n in range(len(src_tr.loss_history)):
            assert src_tr.loss_history[n] == dst_tr.loss_history[n]

        # test loss history
        assert len(src_tr.val_loss_history) == len(dst_tr.val_loss_history)
        assert src_tr.val_loss_iter == dst_tr.val_loss_iter
        for n in range(len(src_tr.val_loss_history)):
            assert src_tr.val_loss_history[n] == dst_tr.val_loss_history[n]

        # test acc history
        assert len(src_tr.acc_history) == len(dst_tr.acc_history)
        assert src_tr.acc_iter == dst_tr.acc_iter
        for n in range(len(src_tr.acc_history)):
            assert src_tr.acc_history[n] == dst_tr.acc_history[n]

        fig, ax = vis_loss_history.get_figure_subplots()
        vis_loss_history.plot_train_history_2subplots(
            ax,
            src_tr.loss_history,
            acc_history = src_tr.acc_history,
            cur_epoch = src_tr.cur_epoch,
            iter_per_epoch = src_tr.iter_per_epoch
        )
        fig.savefig('figures/resnet_trainer_train_test_history.png', bbox_inches='tight')


    def test_train(self) -> None:
        test_checkpoint_name = self.checkpoint_dir + 'resnet_trainer_train_checkpoint.pkl'
        test_history_name    = self.checkpoint_dir + 'resnet_trainer_train_history.pkl'
        train_num_epochs = 4
        train_batch_size = 128
        # get a model
        model = resnets.WideResnet(
            depth = self.resnet_depth,
            num_classes = 10,     # using CIFAR-10 data
            input_channels=3,
            w_factor=1
        )
        # get a traner
        trainer = resnet_trainer.ResnetTrainer(
            model,
            # training parameters
            batch_size    = train_batch_size,
            num_epochs    = train_num_epochs,
            learning_rate = self.test_learning_rate,
            # device
            device_id = util.get_device_id(),
            # checkpoint
            checkpoint_dir = self.checkpoint_dir,
            checkpoint_name = 'resnet_trainer_test',
            # display,
            print_every = self.print_every,
            save_every = 5000,
            verbose = self.verbose
        )

        if self.verbose:
            print('Created %s object' % repr(trainer))
            print(trainer)

        print('Training model %s for %d epochs (batch size = %d)' %\
              (repr(trainer), train_num_epochs, train_batch_size)
        )
        trainer.train()

        # save the final checkpoint
        trainer.save_checkpoint(test_checkpoint_name)
        trainer.save_history(test_history_name)

        fig, ax = vis_loss_history.get_figure_subplots()
        vis_loss_history.plot_train_history_2subplots(
            ax,
            trainer.loss_history,
            acc_history = trainer.acc_history,
            cur_epoch = trainer.cur_epoch,
            iter_per_epoch = trainer.iter_per_epoch
        )
        fig.savefig('figures/resnet_trainer_train_history.png', bbox_inches='tight')
