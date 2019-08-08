"""
TEST_TRAINER
Unit tests for trainer object

Stefan Wong 2018
"""

import os
import sys
import unittest
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
# unit under test
from lernomatic.train import cifar_trainer
from lernomatic.models import cifar
# vis tools
from lernomatic.vis import vis_loss_history

# debug
#from pudb import set_trace; set_trace()


GLOBAL_OPTS = dict()

def get_figure_subplots(num_subplots:int=2) -> tuple:
    fig = plt.figure()
    ax = []
    for p in range(num_subplots):
        sub_ax = fig.add_subplot(num_subplots, 1, (p+1))
        ax.append(sub_ax)

    return (fig, ax)


class TestTrainer(unittest.TestCase):
    def setUp(self):
        self.verbose          = GLOBAL_OPTS['verbose']
        self.draw_plot        = GLOBAL_OPTS['draw_plot']
        self.test_batch_size  = 16
        self.test_num_workers = 2
        self.test_num_epochs  = 2
        self.shuffle          = True
        self.data_dir         = 'data/'

    def test_save_load_checkpoint(self):
        print('======== TestTrainer.test_save_load_checkpoint ')

        test_checkpoint = 'checkpoint/trainer_test_save_load.pkl'

        # get a model
        model = cifar.CIFAR10Net()
        # get a trainer
        test_num_epochs = 1
        src_tr = cifar_trainer.CIFAR10Trainer(
            model,
            num_epochs  = test_num_epochs,
            save_every  = 0,
            device_id   = GLOBAL_OPTS['device_id'],
            batch_size  = self.test_batch_size,
            num_workers = self.test_num_workers
        )

        if self.verbose:
            print('Created trainer object')
            print(src_tr)

        # train for one epoch
        src_tr.train()
        src_tr.save_checkpoint(test_checkpoint)
        # Make a new trainer and load all parameters into that
        # I guess we need to put some kind of loader and model here...
        dst_tr = cifar_trainer.CIFAR10Trainer(
            model,
            device_id = GLOBAL_OPTS['device_id']
        )
        dst_tr.load_checkpoint(test_checkpoint)

        # Test object parameters
        self.assertEqual(src_tr.num_epochs, dst_tr.num_epochs)
        self.assertEqual(src_tr.learning_rate, dst_tr.learning_rate)
        self.assertEqual(src_tr.weight_decay, dst_tr.weight_decay)
        self.assertEqual(src_tr.print_every, dst_tr.print_every)
        self.assertEqual(src_tr.save_every, dst_tr.save_every)
        self.assertEqual(src_tr.device_id, dst_tr.device_id)

        print('\t Comparing model parameters ')
        src_model_params = src_tr.get_model_params()
        dst_model_params = dst_tr.get_model_params()
        self.assertEqual(len(src_model_params.items()), len(dst_model_params.items()))

        # p1, p2 are k,v tuple pairs of each model parameters
        # k = str
        # v = torch.Tensor
        for n, (p1, p2) in enumerate(zip(src_model_params.items(), dst_model_params.items())):
            self.assertEqual(p1[0], p2[0])
            print('Checking parameter %s [%d/%d] \t\t' % (str(p1[0]), n+1, len(src_model_params.items())), end='\r')
            self.assertEqual(True, torch.equal(p1[1], p2[1]))
        print('\n ...done')

        # Test loss history
        val_loss_history = 'test_save_load_history.pkl'
        src_tr.save_history(val_loss_history)
        dst_tr.load_history(val_loss_history)
        print('\t Comparing loss history....')
        self.assertEqual(src_tr.loss_iter, dst_tr.loss_iter)
        for n in range(src_tr.loss_iter):
            print('Checking loss element [%d/%d]' % (n, src_tr.loss_iter), end='\r')
            self.assertEqual(src_tr.loss_history[n], dst_tr.loss_history[n])

        print('\n ...done')
        os.remove(test_checkpoint)

        print('======== TestTrainer.test_save_load_checkpoint <END>')

    def test_save_load_acc(self):
        print('======== TestTrainer.test_save_load_acc ')

        test_checkpoint = 'checkpoint/trainer_test_save_load_acc.pkl'
        test_history = 'checkpoint/trainer_test_save_load_acc_history.pkl'

        model = cifar.CIFAR10Net()
        # Get trainer object
        test_num_epochs = 10
        src_tr = cifar_trainer.CIFAR10Trainer(
            model,
            save_every  = 0,
            print_every = 50,
            device_id   = GLOBAL_OPTS['device_id'],
            # loader options,
            num_epochs  = self.test_num_epochs,
            batch_size  = self.test_batch_size,
            num_workers = self.test_num_workers,
        )

        if self.verbose:
            print('Created trainer object')
            print(src_tr)

        # train for one epoch
        src_tr.train()
        src_tr.save_checkpoint(test_checkpoint)
        self.assertIsNot(None, src_tr.acc_history)
        src_tr.save_history(test_history)

        # Now try to load a checkpoint and ensure that there is an
        # acc history attribute that is not None
        dst_tr = cifar_trainer.CIFAR10Trainer(
            model,
            device_id = GLOBAL_OPTS['device_id'],
            verbose = self.verbose
        )
        dst_tr.load_checkpoint(test_checkpoint)

        # Test object parameters
        self.assertEqual(src_tr.num_epochs, dst_tr.num_epochs)
        self.assertEqual(src_tr.learning_rate, dst_tr.learning_rate)
        self.assertEqual(src_tr.weight_decay, dst_tr.weight_decay)
        self.assertEqual(src_tr.print_every, dst_tr.print_every)
        self.assertEqual(src_tr.save_every, dst_tr.save_every)
        self.assertEqual(src_tr.device_id, dst_tr.device_id)

        print('\t Comparing model parameters ')
        src_model_params = src_tr.get_model_params()
        dst_model_params = dst_tr.get_model_params()
        self.assertEqual(len(src_model_params.items()), len(dst_model_params.items()))

        print('\t Comparing model parameters ')
        src_model_params = src_tr.get_model_params()
        dst_model_params = dst_tr.get_model_params()
        self.assertEqual(len(src_model_params.items()), len(dst_model_params.items()))

        # p1, p2 are k,v tuple pairs of each model parameters
        # k = str
        # v = torch.Tensor
        for n, (p1, p2) in enumerate(zip(src_model_params.items(), dst_model_params.items())):
            self.assertEqual(p1[0], p2[0])
            print('Checking parameter %s [%d/%d] \t\t' % (str(p1[0]), n, len(src_model_params.items())), end='\r')
            self.assertEqual(True, torch.equal(p1[1], p2[1]))
        print('\n ...done')

        # Test loss history
        dst_tr.load_history(test_history)
        self.assertIsNot(None, dst_tr.acc_history)

        print('\t Comparing loss history....')
        self.assertEqual(src_tr.loss_iter, dst_tr.loss_iter)
        for n in range(src_tr.loss_iter):
            print('Checking loss element [%d/%d]' % (n, src_tr.loss_iter), end='\r')
            self.assertEqual(src_tr.loss_history[n], dst_tr.loss_history[n])

        os.remove(test_checkpoint)
        os.remove(test_history)

        print('======== TestTrainer.test_save_load_acc <END>')

    def test_save_load_device_map(self):
        print('======== TestTrainer.test_save_load_device_map ')

        test_checkpoint = 'checkpoint/trainer_save_load_device_map.pkl'
        test_history = 'checkpoint/trainer_save_load_device_map_history.pkl'

        model = cifar.CIFAR10Net()
        # Get trainer object
        test_num_epochs = 10
        src_tr = cifar_trainer.CIFAR10Trainer(
            model,
            save_every  = 0,
            print_every = 50,
            device_id   = GLOBAL_OPTS['device_id'],
            # loader options,
            num_epochs  = self.test_num_epochs,
            batch_size  = self.test_batch_size,
            num_workers = self.test_num_workers,
        )

        if self.verbose:
            print('Created trainer object')
            print(src_tr)

        # train for one epoch
        src_tr.train()
        src_tr.save_checkpoint(test_checkpoint)
        self.assertIsNot(None, src_tr.acc_history)
        src_tr.save_history(test_history)

        # Now try to load a checkpoint and ensure that there is an
        # acc history attribute that is not None
        dst_tr = cifar_trainer.CIFAR10Trainer(
            model,
            device_id = -1,
            verbose = self.verbose
        )
        dst_tr.load_checkpoint(test_checkpoint)

        # Test object parameters
        self.assertEqual(src_tr.num_epochs, dst_tr.num_epochs)
        self.assertEqual(src_tr.learning_rate, dst_tr.learning_rate)
        self.assertEqual(src_tr.weight_decay, dst_tr.weight_decay)
        self.assertEqual(src_tr.print_every, dst_tr.print_every)
        self.assertEqual(src_tr.save_every, dst_tr.save_every)

        print('\t Comparing model parameters ')
        src_model_params = src_tr.get_model_params()
        dst_model_params = dst_tr.get_model_params()
        self.assertEqual(len(src_model_params.items()), len(dst_model_params.items()))

        print('\t Comparing model parameters ')
        src_model_params = src_tr.get_model_params()
        dst_model_params = dst_tr.get_model_params()
        self.assertEqual(len(src_model_params.items()), len(dst_model_params.items()))

        # p1, p2 are k,v tuple pairs of each model parameters
        # k = str
        # v = torch.Tensor
        print('Checking that tensors from checkpoint have been tranferred to new device')
        for n, (p1, p2) in enumerate(zip(src_model_params.items(), dst_model_params.items())):
            self.assertEqual(p1[0], p2[0])
            print('Checking parameter %s [%d/%d] \t\t' % (str(p1[0]), n, len(src_model_params.items())), end='\r')
            self.assertEqual('cpu', p2[1].device.type)
        print('\n ...done')

        # Test loss history
        dst_tr.load_history(test_history)
        self.assertIsNot(None, dst_tr.acc_history)
        print('\t Comparing loss history....')
        self.assertEqual(src_tr.loss_iter, dst_tr.loss_iter)
        for n in range(src_tr.loss_iter):
            print('Checking loss element [%d/%d]' % (n, src_tr.loss_iter), end='\r')
            self.assertEqual(src_tr.loss_history[n], dst_tr.loss_history[n])
        print('======== TestTrainer.test_save_load_device_map <END>')

        os.remove(test_checkpoint)
        os.remove(test_history)

        print('======== TestTrainer.test_save_load_acc <END>')

    def test_save_load_device_map(self):
        print('======== TestTrainer.test_save_load_device_map ')

        test_checkpoint = 'checkpoint/trainer_save_load_device_map.pkl'
        test_history = 'checkpoint/trainer_save_load_device_map_history.pkl'

        model = cifar.CIFAR10Net()
        # Get trainer object
        test_num_epochs = 10
        src_tr = cifar_trainer.CIFAR10Trainer(
            model,
            save_every  = 0,
            print_every = 50,
            device_id   = GLOBAL_OPTS['device_id'],
            # loader options,
            num_epochs  = self.test_num_epochs,
            batch_size  = self.test_batch_size,
            num_workers = self.test_num_workers,
        )

        if self.verbose:
            print('Created trainer object')
            print(src_tr)

        # train for one epoch
        src_tr.train()
        src_tr.save_checkpoint(test_checkpoint)
        self.assertIsNot(None, src_tr.acc_history)
        src_tr.save_history(test_history)

        # Now try to load a checkpoint and ensure that there is an
        # acc history attribute that is not None
        dst_tr = cifar_trainer.CIFAR10Trainer(
            model,
            device_id = -1,
            verbose = self.verbose
        )
        dst_tr.load_checkpoint(test_checkpoint)

        # Test object parameters
        self.assertEqual(src_tr.num_epochs, dst_tr.num_epochs)
        self.assertEqual(src_tr.learning_rate, dst_tr.learning_rate)
        self.assertEqual(src_tr.weight_decay, dst_tr.weight_decay)
        self.assertEqual(src_tr.print_every, dst_tr.print_every)
        self.assertEqual(src_tr.save_every, dst_tr.save_every)

        print('\t Comparing model parameters ')
        src_model_params = src_tr.get_model_params()
        dst_model_params = dst_tr.get_model_params()
        self.assertEqual(len(src_model_params.items()), len(dst_model_params.items()))

        print('\t Comparing model parameters ')
        src_model_params = src_tr.get_model_params()
        dst_model_params = dst_tr.get_model_params()
        self.assertEqual(len(src_model_params.items()), len(dst_model_params.items()))

        # p1, p2 are k,v tuple pairs of each model parameters
        # k = str
        # v = torch.Tensor
        print('Checking that tensors from checkpoint have been tranferred to new device')
        for n, (p1, p2) in enumerate(zip(src_model_params.items(), dst_model_params.items())):
            self.assertEqual(p1[0], p2[0])
            print('Checking parameter %s [%d/%d] \t\t' % (str(p1[0]), n, len(src_model_params.items())), end='\r')
            self.assertEqual('cpu', p2[1].device.type)
        print('\n ...done')

        # Test loss history
        dst_tr.load_history(test_history)
        self.assertIsNot(None, dst_tr.acc_history)
        print('\t Comparing loss history....')
        self.assertEqual(src_tr.loss_iter, dst_tr.loss_iter)
        for n in range(src_tr.loss_iter):
            print('Checking loss element [%d/%d]' % (n, src_tr.loss_iter), end='\r')
            self.assertEqual(src_tr.loss_history[n], dst_tr.loss_history[n])
        print('======== TestTrainer.test_save_load_device_map <END>')

    def test_train(self):
        print('======== TestTrainer.test_train ')

        test_checkpoint = 'checkpoint/trainer_train_test.pkl'
        model = cifar.CIFAR10Net()

        # Get trainer object
        trainer = cifar_trainer.CIFAR10Trainer(
            model,
            save_every    = 0,
            print_every   = 50,
            device_id     = GLOBAL_OPTS['device_id'],
            # loader options,
            num_epochs    = self.test_num_epochs,
            learning_rate = 3e-4,
            batch_size    = 128,
            num_workers   = self.test_num_workers,
        )

        if self.verbose:
            print('Created trainer object')
            print(trainer)

        # train for one epoch
        trainer.train()
        trainer.save_checkpoint(test_checkpoint)

        fig, ax = get_figure_subplots()
        vis_loss_history.plot_train_history_2subplots(
            ax,
            trainer.loss_history,
            acc_history = trainer.acc_history,
            cur_epoch = trainer.cur_epoch,
            iter_per_epoch = trainer.iter_per_epoch,
            loss_title = 'CIFAR-10 Trainer Test loss',
            acc_title = 'CIFAR-10 Trainer Test accuracy'
        )
        fig.savefig('figures/trainer_train_test_history.png', bbox_inches='tight')

        print('======== TestTrainer.test_train <END>')

    def test_history_extend(self):
        print('======== TestTrainer.test_history_extend ')

        test_checkpoint = 'checkpoint/test_history_extend.pkl'
        test_history = 'checkpoint/test_history_extend_history.pkl'
        test_num_epochs = 4
        model = cifar.CIFAR10Net()
        # Get trainer object
        test_num_epochs = 10
        trainer = cifar_trainer.CIFAR10Trainer(
            model,
            save_every    = 0,
            print_every   = 50,
            device_id     = GLOBAL_OPTS['device_id'],
            # loader options,
            num_epochs    = test_num_epochs,
            learning_rate = 3e-4,
            batch_size    = 64,
            num_workers   = self.test_num_workers,
        )
        print('Training original model')
        trainer.train()
        trainer.save_checkpoint(test_checkpoint)
        trainer.save_history(test_history)

        # Load a new trainer, train for another 10 epochs (20 total)
        extend_trainer = cifar_trainer.CIFAR10Trainer(
            model,
            save_every    = 0,
            print_every   = 50,
            device_id     = GLOBAL_OPTS['device_id'],
            # loader options,
            num_epochs    = 10,
            learning_rate = 3e-4,
            batch_size    = 64,
            num_workers   = self.test_num_workers,
        )
        print('Loading checkpoint [%s] into extend trainer...' % str(test_checkpoint))
        extend_trainer.load_checkpoint(test_checkpoint)
        print('Loading history [%s] into extend trainer...' % str(test_history))
        extend_trainer.load_history(test_history)
        # Check history before extending
        self.assertEqual(trainer.device_id, extend_trainer.device_id)
        print('extend_trainer device : %s' % str(extend_trainer.device))
        self.assertEqual(test_num_epochs, extend_trainer.num_epochs)
        self.assertEqual(10, extend_trainer.cur_epoch)
        self.assertIsNot(None, extend_trainer.loss_history)
        self.assertEqual(10 * len(extend_trainer.train_loader), len(extend_trainer.loss_history))

        extend_trainer.set_num_epochs(20)
        self.assertEqual(20 * len(extend_trainer.train_loader), len(extend_trainer.loss_history))
        for i in range(10 * len(extend_trainer.train_loader)):
            print('Checking loss iter [%d / %d]' % (i, 20 * len(extend_trainer.train_loader)), end='\r')
            self.assertEqual(trainer.loss_history[i], extend_trainer.loss_history[i])

        extend_trainer.train()

        print('======== TestTrainer.test_history_extend <END>')


# Entry point
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose',
                        action='store_true',
                        default=False,
                        help='Sets verbose mode'
                        )
    parser.add_argument('--draw-plot',
                        action='store_true',
                        default=False,
                        help='Draw plots'
                        )
    parser.add_argument('--num-workers',
                        type=int,
                        default=1,
                        help='Number of worker processes to use for HDF5 load'
                        )
    parser.add_argument('--batch-size',
                        type=int,
                        default=8,
                        help='Batch size to use during training'
                        )
    parser.add_argument('--device-id',
                        type=int,
                        default=-1,
                        help='Device to use for tests (default : -1)'
                        )
    parser.add_argument('unittest_args', nargs='*')

    args = parser.parse_args()
    arg_vals = vars(args)
    for k, v in arg_vals.items():
        GLOBAL_OPTS[k] = v

    if GLOBAL_OPTS['verbose']:
        print('-------- GLOBAL OPTS (%s) --------' % str(sys.argv[0]))
        for k, v in GLOBAL_OPTS.items():
            print('\t[%s] : %s' % (str(k), str(v)))


    sys.argv[1:] = args.unittest_args
    unittest.main()
