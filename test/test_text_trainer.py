"""
TEST_TEXT_TRAINER
Unit tests for TextTrainer object

Stefan Wong 2019
"""


import sys
import argparse
import unittest
import torch
import matplotlib.pyplot as plt
# units under test
from lernomatic.train import text_trainer
from lernomatic.models import text
from lernomatic.vis import vis_loss_history

# debug
#from pudb import set_trace; set_trace()

GLOBAL_OPTS = dict()


def get_text_trainer(word_map):

    # TODO : get data into test


    model = text.TextRNN(
        len(word_map)
    )

    trainer = text_trainer.TextTrainer(
        model,
        # word map
        word_map = word_map,
        # datasets

        # training options
        num_epochs = GLOBAL_OPTS['num_epochs'],
        learning_rate = GLOBAL_OPTS['learning_rate'],

        verbose = GLOBAL_OPTS['verbose']

    )

    return trainer



class TestTextTrainer(unittest.TestCase):
    def setUp(self):
        self.verbose = GLOBAL_OPTS['verbose']
        self.test_num_epochs = 8

        # TODO : need to generate wordmap here

    def test_save_load_checkpoint(self):
        print('======== TestTextTrainer.test_save_load_checkpoint ')

        src_tr_checkpoint_file = 'checkpoint/test_text_trainer.pkl'
        src_tr_history_file = 'checkpoint/test_text_trainer_history.pkl'

        src_tr = get_text_trainer()
        src_tr.save_every = 0      # we just save a single checkpoint at the end
        src_tr.train()
        src_tr.save_checkpoint(src_tr_checkpoint_file)
        src_tr.save_history(src_tr_history_file)

        dst_tr = get_text_trainer()
        dst_tr.load_checkpoint(src_tr_checkpoint_file)

        # check parameters
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

        dst_tr.load_history(test_history_name)

        # loss history
        self.assertEqual(len(src_tr.loss_history), len(dst_tr.loss_history))
        self.assertEqual(src_tr.loss_iter, dst_tr.loss_iter)
        for n in range(len(src_tr.loss_history)):
            self.assertEqual(src_tr.loss_history[n], dst_tr.loss_history[n])

        # test loss history
        self.assertEqual(len(src_tr.test_loss_history), len(dst_tr.test_loss_history))
        self.assertEqual(src_tr.test_loss_iter, dst_tr.test_loss_iter)
        for n in range(len(src_tr.test_loss_history)):
            self.assertEqual(src_tr.test_loss_history[n], dst_tr.test_loss_history[n])

        # test acc history
        self.assertEqual(len(src_tr.acc_history), len(dst_tr.acc_history))
        self.assertEqual(src_tr.acc_iter, dst_tr.acc_iter)
        for n in range(len(src_tr.acc_history)):
            self.assertEqual(src_tr.acc_history[n], dst_tr.acc_history[n])

        fig, ax = vis_loss_history.get_figure_subplots()
        vis_loss_history.plot_train_history_2subplots(
            ax,
            src_tr.loss_history,
            acc_history = src_tr.acc_history,
            cur_epoch = src_tr.cur_epoch,
            iter_per_epoch = src_tr.iter_per_epoch
        )
        fig.savefig('figures/text_trainer_train_test_history.png', bbox_inches='tight')

        print('======== TestTextTrainer.test_save_load_checkpoint <END>')


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
    # dataset options
    parser.add_argument('--train-dataset',
                        type=str,
                        default='hdf5/test_text_train.h5',
                        help='File to load train data from'
                        )
    parser.add_argument('--test-dataset',
                        type=str,
                        default='hdf5/test_text_test.h5',
                        help='File to load test data from'
                        )
    parser.add_argument('--val-dataset',
                        type=str,
                        default='hdf5/test_text_val.h5',
                        help='File to load val data from'
                        )


    # display options
    parser.add_argument('--print-every',
                        type=int,
                        default=100,
                        help='Print output every time this number of iterations has elapsed'
                        )
    # checkpoint options
    parser.add_argument('--checkpoint-dir',
                        type=str,
                        default='checkpoint/',
                        help='Directory to save checkpoints to'
                        )
    parser.add_argument('--checkpoint-name',
                        type=str,
                        default='resnet-trainer-test',
                        help='String to prefix to checkpoint files'
                        )
    parser.add_argument('unittest_args', nargs='*')

    args = parser.parse_args()
    arg_vals = vars(args)
    for k, v in arg_vals.items():
        GLOBAL_OPTS[k] = v

    if GLOBAL_OPTS['verbose']:
        print('-------- GLOBAL OPTS (%s) --------' % str(sys.argv[0]))
        for k, v in GLOBAL_OPTS.items():
            print('[%s] : %s' % (str(k), str(v)))


    sys.argv[1:] = args.unittest_args
    unittest.main()
