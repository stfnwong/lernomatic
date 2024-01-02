""""
TEST_IMAGE_PROC
Unit tests for Image Proc module. Because torchvision requires PIL it turns out
to be handy to be able to specify whether or not we want the output to be an
Numpy ndarray or a PIL Image.

Stefan Wong 2019
"""

import os
import cv2
import PIL
import numpy as np
import pytest

# unit(s) under test
from lernomatic.data import image_proc
from lernomatic.data import data_split
from lernomatic.data import hdf5_dataset

from typing import Tuple


# Helper function to check files
def check_files(img_paths:list) -> Tuple[list, int]:
    num_err = 0
    for n, path in enumerate(img_paths):
        print('Checking file [%d / %d] ' % (n+1, len(img_paths)), end='\r')

        # If there are any exceptions then just remove the file that caused
        # them and continue
        img = cv2.imread(path)
        if img is None:
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


class TestImageProc:
    # TODO : make this settable...
    dataset_root    = '/mnt/ml-data/datasets/cyclegan/night2day/val/'
    verbose         = True
    test_image_size = 128
    # output files
    ndarray_test_outfile = 'hdf5/image_proc_ndarray_test.h5'
    pil_test_outfile     = 'hdf5/image_proc_pil_test.h5'
    # expected output shapes
    expected_ndarray_shape = (3, 128, 128)
    expected_pil_shape     = (128, 128, 3)

    @pytest.mark.skip(reason='Need a way to test this without relying on my data path')
    def test_img_ndarray(self) -> None:
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
        assert proc.to_pil is False
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
            assert self.expected_ndarray_shape == image.shape
            assert isinstance(image, np.ndarray) is True
        print('\n OK')


    @pytest.mark.skip(reason='Need a way to test this without relying on my data path')
    def test_img_pil(self) -> None:
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
        assert proc.to_pil is True
        proc.proc(s, self.pil_test_outfile)

        print('Processed file [%s], checking...' % str(self.pil_test_outfile))
        # Now read back the data and check
        test_data = hdf5_dataset.HDF5PILDataset(
            self.pil_test_outfile,
            feature_name = 'images',
            label_name = 'labels'
        )

        # NOTE : seems that we cannot store PIL image in HDF5 directly, rather
        # we ought to convert when reading...
        for idx, (image, label) in enumerate(test_data):
            print('Checking element [%d / %d]' % (idx+1, len(test_data)), end='\r')
            assert self.expected_pil_shape[0:2] == image.size
            assert isinstance(image, PIL.Image.Image) is True
        print('\n OK')
