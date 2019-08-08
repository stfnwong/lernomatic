""""
TEST_IMAGE_PROC
Unit tests for Image Proc module. Because torchvision requires PIL it turns out
to be handy to be able to specify whether or not we want the output to be an
Numpy ndarray or a PIL Image.

Stefan Wong 2019
"""

import sys
import os
import argparse
import unittest
import torch
import cv2
import PIL
import numpy as np
# unit(s) under test
from lernomatic.data import image_proc
from lernomatic.data import data_split
from lernomatic.data import hdf5_dataset

from typing import Tuple

# debug
#from pudb import set_trace; set_trace()


GLOBAL_OPTS = dict()


# Helper function to check files
def check_files(img_paths:list) -> Tuple[list, int]:
    num_err = 0
    for n, path in enumerate(img_paths):
        if GLOBAL_OPTS['verbose']:
            print('Checking file [%d / %d] ' % (n+1, len(img_paths)), end='\r')

        # If there are any exceptions then just remove the file that caused
        # them and continue
        img = cv2.imread(path)
        if img is None:
            if GLOBAL_OPTS['verbose']:
                print('Failed to load image [%d/%d] <%s>' % (n, len(img_paths), str(path)))
            img_paths.pop(n)
            num_err += 1

    return (img_paths, num_err)


# helper function to get a split object
def get_split(img_paths:str, split_name:str) -> data_split.DataSplit:
    s = data_split.DataSplit(split_name=split_name)
    s.data_paths  = img_paths
    s.data_labels = [int(0) for _ in range(len(img_paths))]
    s.elem_ids    = [int(0) for _ in range(len(img_paths))]
    s.has_labels  = True
    s.has_ids     = True

    return s


class TestImageProc(unittest.TestCase):
    def setUp(self):
        self.dataset_root    = GLOBAL_OPTS['dataset_root']
        self.verbose         = GLOBAL_OPTS['verbose']
        self.test_image_size = 128
        # output files
        self.ndarray_test_outfile = 'hdf5/image_proc_ndarray_test.h5'
        self.pil_test_outfile = 'hdf5/image_proc_pil_test.h5'
        # expected output shapes
        self.expected_ndarray_shape = (3, 128, 128)
        self.expected_pil_shape     = (128, 128, 3)

    def test_img_ndarray(self):
        print('======== TestImageProc.test_img_ndarray ')

        raw_img_paths = [self.dataset_root + str(path) for path in os.listdir(self.dataset_root)]
        print('Found %d files in directory [%s]' % (len(raw_img_paths), str(self.dataset_root)))
        img_paths, num_invalid = check_files(raw_img_paths)

        if num_invalid > 0:
            print('Found %d invalid images in directory [%s]' % (num_invalid, str(self.dataset_root)))

        s = get_split(img_paths, 'ndarray_test')

        # process the data
        proc = image_proc.ImageDataProc(
            label_dataset_size = 1,
            image_dataset_size = (3, self.test_image_size, self.test_image_size),
            to_pil = False,
            verbose = self.verbose
        )
        self.assertEqual(False, proc.to_pil)
        proc.proc(s, self.ndarray_test_outfile)

        print('Processed file [%s], checking...' % str(self.ndarray_test_outfile))
        # Now read back the data and check
        test_data = hdf5_dataset.HDF5RawDataset(
            self.ndarray_test_outfile,
            feature_name = 'images',
            label_name = 'labels'
        )

        for idx, (image, label) in enumerate(test_data):
            print('Checking element [%d / %d]' % (idx+1, len(test_data)), end='\r')
            self.assertEqual(self.expected_ndarray_shape, image.shape)
            self.assertTrue(isinstance(image, np.ndarray))

        print('\n OK')

        print('======== TestImageProc.test_img_ndarray <END>')

    def test_img_pil(self):
        print('======== TestImageProc.test_img_pil ')

        #raw_img_paths = os.listdir(self.dataset_root)
        raw_img_paths = [self.dataset_root + str(path) for path in os.listdir(self.dataset_root)]
        print('Found %d files in directory [%s]' % (len(raw_img_paths), str(self.dataset_root)))
        img_paths, num_invalid = check_files(raw_img_paths)

        if num_invalid > 0:
            print('Found %d invalid images in directory [%s]' % (num_invalid, str(self.dataset_root)))

        s = get_split(img_paths, 'pil_test')

        # NOTE: because this isa PIL image, we need to ensure that the dataset
        # shape has the number of channels at the end of the tuple
        proc = image_proc.ImageDataProc(
            label_dataset_size = 1,
            image_dataset_size = (self.test_image_size, self.test_image_size,3),
            to_pil = True,
            verbose = self.verbose
        )
        self.assertEqual(True, proc.to_pil)
        proc.proc(s, self.pil_test_outfile)

        print('Processed file [%s], checking...' % str(self.pil_test_outfile))
        # Now read back the data and check
        test_data = hdf5_dataset.HDF5RawDataset(
            self.pil_test_outfile,
            feature_name = 'images',
            label_name = 'labels'
        )

        # NOTE : seems that we cannot store PIL image in HDF5 directly, rather
        # we ought to convert when reading...
        for idx, (image, label) in enumerate(test_data):
            print('Checking element [%d / %d]' % (idx+1, len(test_data)), end='\r')
            self.assertEqual(self.expected_pil_shape, image.shape)
            #self.assertTrue(isinstance(image, PIL.Image.Image))

        print('\n OK')

        print('======== TestImageProc.test_img_pil <END>')


# Entry point
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose',
                        action='store_true',
                        default=False,
                        help='Sets verbose mode'
                        )
    # root of data for test
    parser.add_argument('--dataset-root',
                        type=str,
                        default='/mnt/ml-data/datasets/cyclegan/night2day/val/',
                        help='Path to root of dataset to use for test'
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
