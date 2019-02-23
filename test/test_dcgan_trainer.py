"""
TEST_DCGAN_TRAINER
Unit tests for DCGATrainer module

Stefan Wong 2019
"""

import sys
import argparse
import unittest
import torch
# units under test
from lernomatic.model import dcgan
from lernomatic.train import trainer


# debug
from pudb import set_trace; set_trace()

GLOBAL_OPTS = dict()


def get_dataset():
    celeba_transform = transforms.Compose([
           transforms.Resize(GLOBAL_OPTS['image_size']),
           transforms.CenterCrop(image_size),
           transforms.ToTensor(),
           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = dset.ImageFolder(
        root=GLOBAL_OPTS['dataset_root'],
        transform = celeba_transform
    )

    return dataset


class TestDCGANTrainer(unittest.TestCase):
    def setUp(self):
        self.verbose = GLOBAL_OPTS['verbose']

    def test_save_load_checkpoint(self):
        print('======== TestDCGANTrainer.test_save_load_checkpoint ')

        # TODO : need dataloaders

        discriminator = dcgan.DCGDiscriminator()
        generator = dcgan.DCGGenerator()

        trainer = dcgan_trainer.DCGANTrainer(
            D = discriminator,
            G = generator,
            # training params
            num_epochs = GLOBAL_OPTS['num_epochs'],
            learning_rate = GLOBAL_OPTS['learning_rate']
        )

        print('======== TestDCGANTrainer.test_save_load_checkpoint <END>')



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
    # display options
    parser.add_argument('--print-every',
                        type=int,
                        default=100,
                        help='Print output every time this number of iterations has elapsed'
                        )
    # dataset options
    parser.add_argument('--dataset-root',
                        type=str,
                        default='/mnt/ml-data/datasets/celeba/img_align_celeba/',
                        help='Path to root of dataset'
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
