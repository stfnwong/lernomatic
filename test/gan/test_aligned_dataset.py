"""
TEST_ALIGNED_DATASET
Unit tests for the aligned dataset

Stefan Wong 2019
"""

import os
import sys
import unittest
import argparse
import numpy as np
import h5py
# unit(s) under test
from lernomatic.data.gan import aligned_data_split
from lernomatic.data.gan import aligned_data_proc
from lernomatic.data.gan import aligned_dataset
from lernomatic.data import hdf5_dataset
from lernomatic.util import file_util

# debug
from pudb import set_trace; set_trace()


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


class TestAlignedDataSplit(unittest.TestCase):
    def setUp(self):
        self.test_a_path = 'testA/'
        self.test_b_path = 'testB/'
        self.test_data_root = '/home/kreshnik/ml-data/monet2photo/'  # TODO: make settable
        # number of path pairs to place into split
        self.split_size  = 100
        self.verbose     = GLOBAL_OPTS['verbose']

    def test_save_load(self):
        print('======== TestAlignedDataSplit.test_save_load ')

        test_a_paths, test_b_paths = gen_test_file_lists(
            self.test_data_root,
            self.test_a_path,
            self.test_b_path
        )
        # create a new data split
        data_split = aligned_data_split.AlignedDataSplit(
            split_name = 'test_data_split'
        )

        for pair_idx, (a_path, b_path) in enumerate(zip(test_a_paths, test_b_paths)):
            if self.verbose:
                print('Adding path pair [%d / %d]' % (pair_idx+1, self.split_size), end='\r')
            data_split.add_paths(a_path, b_path)
            if pair_idx >= self.split_size-1:
                break;

        if self.verbose:
            print('\n OK')

        # This is actually a pretty weak test so far... more to come
        self.assertEqual(self.split_size, len(data_split))
        self.assertEqual(False, data_split.has_ids)

        print('======== TestAlignedDataSplit.test_save_load <END>')



#class TestAlignedImageJoin(unittest.TestCase):
#    def setUp(self):
#        self.test_a_path = 'testA/'
#        self.test_b_path = 'testB/'
#        self.test_data_root = '/home/kreshnik/ml-data/monet2photo/'  # TODO: make settable
#        # number of path pairs to place into split
#        self.split_size  = 1024
#        self.verbose     = GLOBAL_OPTS['verbose']
#        # image properties
#        self.test_image_shape = (3, 256, 256)



class TestAlignedImageSplit(unittest.TestCase):
    def setUp(self):
        self.test_data_root = "/mnt/ml-data/datasets/cyclegan/night2day/train/"
        self.split_size     = 1024
        self.verbose        = GLOBAL_OPTS['verbose']
        # image properties
        self.test_image_shape = (3, 256, 256)

    def test_init(self):
        print('======== TestAlignedImageSplit.test_init ')

        dataset_paths = file_util.get_file_paths(
            self.test_data_root,
            verbose = self.verbose
        )
        print('Processing %d images starting at root path [%s]' %\
              (len(dataset_paths), str(self.test_data_root))
        )
        self.assertNotEqual(0, len(dataset_paths))

        # get a data processor
        test_image_dataset_name = 'images'
        align_proc = aligned_data_proc.AlignedImageSplit(
            image_dataset_name = test_image_dataset_name,
            image_shape = self.test_image_shape,
            verbose = self.verbose
        )
        #self.assertEqual('AB', align_proc.direction)

        test_outfile = 'data/test_aligned_proc.h5'
        align_proc.proc(dataset_paths, test_outfile)

        # Test as raw *.h5 file
        with h5py.File(test_outfile, 'r') as fp:
            dataset_keys = fp.keys()
            self.assertIn('a_imgs', dataset_keys)
            self.assertIn('b_imgs', dataset_keys)

        ## Test as HDF5Dataset
        test_dataset = aligned_dataset.AlignedDatasetHDF5(
            test_outfile,
            image_dataset_name = align_proc.image_dataset_name,
            a_img_name = align_proc.a_img_name,
            b_img_name = align_proc.b_img_name,
            verbose = self.verbose
        )
        self.assertEqual(len(dataset_paths), len(test_dataset))

        # Check that the elements in the dataset are the expected size
        for elem_idx, (a_img, b_img) in enumerate(test_dataset):
            print('Checking element [%d / %d]' % (elem_idx+1, len(test_dataset)), end='\r')
            self.assertEqual(self.test_image_shape, a_img.shape)
            self.assertEqual(self.test_image_shape, b_img.shape)

        print('\n OK')

        # now that the test is over, get rid of the file
        os.remove(test_outfile)

        print('======== TestAlignedImageSplit.test_init <END>')



# Entry point
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose',
                        action='store_true',
                        default=False,
                        help='Sets verbose mode'
                        )
    # data paths
    parser.add_argument('--test-data-root',
                        type=str,
                        default='/home/kreshnik/ml-data/monet2photo',
                        help='Path to root of test data'
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
