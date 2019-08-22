"""
TEST_UNALIGNED_DATASET
Unit tests for un-aligned dataset

Stefan Wong 2019
"""

import os
import sys
import unittest
import argparse
import numpy as np
import h5py
# unit(s) under test
from lernomatic.data.gan import unaligned_data_proc
from lernomatic.data.gan import unaligned_dataset
from lernomatic.data import hdf5_dataset
from lernomatic.util import file_util


GLOBAL_OPTS = dict()


def gen_test_file_lists(data_root:str, test_a_path:str, test_b_path:str, crop_len:bool=True) -> tuple:

    test_a_root = os.path.join(data_root, test_a_path)
    test_b_root = os.path.join(data_root, test_b_path)
    test_a_paths = [test_a_root + path for path in os.listdir(test_a_root)]
    test_b_paths = [test_b_root + path for path in os.listdir(test_b_root)]

    if crop_len:
        if len(test_a_paths) > len(test_b_paths):
            test_a_paths = test_a_paths[0 : len(test_b_paths)]
        elif len(test_b_paths) > len(test_a_paths):
            test_b_paths = test_b_paths[0 : len(test_a_paths)]

    return (test_a_paths, test_b_paths)


class TestUnalignedDataset(unittest.TestCase):
    def setUp(self):
        self.test_a_path = 'testA/'
        self.test_b_path = 'testB/'
        self.test_data_root = '/mnt/ml-data/datasets/cyclegan/monet2photo/'
        # number of path pairs to place into split
        self.split_size  = 100
        self.verbose     = GLOBAL_OPTS['verbose']



class TestUnalignedImageProc(unittest.TestCase):
    def setUp(self):
        self.test_a_path = 'testA/'
        self.test_b_path = 'testB/'
        self.test_data_root = '/mnt/ml-data/datasets/cyclegan/monet2photo/'
        # number of path pairs to place into split
        self.split_size  = 4096
        self.verbose     = GLOBAL_OPTS['verbose']
        # image properties
        self.test_image_shape = (3, 256, 256)


    def test_proc(self):
        print('======== TestUnalignedImageProc.test_proc ')

        # Get some paths
        test_a_paths = file_util.get_file_paths(
            str(self.test_data_root) + str(self.test_a_path),
            verbose = self.verbose
        )
        self.assertNotEqual(0, len(test_a_paths))

        test_b_paths = file_util.get_file_paths(
            str(self.test_data_root) + str(self.test_b_path),
            verbose = self.verbose
        )
        self.assertNotEqual(0, len(test_b_paths))

        print('Processing %d A images, %d B images from paths [%s]. [%s]' %\
              (len(test_a_paths), len(test_b_paths),
               str(self.test_data_root) + str(self.test_a_path),
               str(self.test_data_root) + str(self.test_b_path))
        )

        test_image_dataset_name = 'images'
        unalign_proc = unaligned_data_proc.UnalignedImageProc(
            image_dataset_name = test_image_dataset_name,
            image_shape        = self.test_image_shape,
            verbose            = self.verbose
        )
        test_outfile = 'data/test_unaligned_image_proc.h5'
        unalign_proc.proc(test_a_paths, test_b_paths, test_outfile)

        # Now load the data into a HDF5 Dataset and check keys
        test_dataset = unaligned_dataset.UnalignedDatasetHDF5(
            test_outfile
        )
        self.assertEqual(len(test_a_paths), test_dataset.get_a_dataset_len())
        self.assertEqual(len(test_b_paths), test_dataset.get_b_dataset_len())

        for elem_idx, (a_img, b_img) in enumerate(test_dataset):
            print('Checking element [%d / %d]' % (elem_idx+1, len(test_dataset)), end='\r')
            self.assertEqual(self.test_image_shape, a_img.shape)
            self.assertEqual(self.test_image_shape, b_img.shape)


        print('======== TestUnalignedImageProc.test_proc <END>')



# Entry point
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose',
                        action='store_true',
                        default=False,
                        help='Sets verbose mode'
                        )
    parser.add_argument('--remove',
                        action='store_true',
                        default=False,
                        help='Remove generated files at the end of tests'
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
