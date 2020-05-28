"""
TEST_UNALIGNED_DATASET
Unit tests for un-aligned dataset

Stefan Wong 2019
"""

import pytest
import numpy as np
import h5py
# unit(s) under test
from lernomatic.data.gan import unaligned_data_proc
from lernomatic.data.gan import unaligned_dataset
from lernomatic.data import hdf5_dataset
from lernomatic.util import file_util
from test import util


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


class TestUnalignedDataset:
    test_a_path = 'testA/'
    test_b_path = 'testB/'
    test_data_root = '/mnt/ml-data/datasets/cyclegan/monet2photo/'
    # number of path pairs to place into split
    split_size  = 100
    verbose     = True



class TestUnalignedImageProc:
    test_a_path = 'testA/'
    test_b_path = 'testB/'
    test_data_root = '/mnt/ml-data/datasets/cyclegan/monet2photo/'
    # number of path pairs to place into split
    split_size  = 4096
    verbose     = True
    # image properties
    test_image_shape = (3, 256, 256)

    @pytest.mark.skip(reason='Need a way to test this without relying on my data path')
    def test_proc(self) -> None:
        # Get some paths
        test_a_paths = file_util.get_file_paths(
            str(self.test_data_root) + str(self.test_a_path),
            verbose = self.verbose
        )
        assert len(test_a_paths) != 0

        test_b_paths = file_util.get_file_paths(
            str(self.test_data_root) + str(self.test_b_path),
            verbose = self.verbose
        )
        assert len(test_b_paths) != 0

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
        assert len(test_a_paths) == test_dataset.get_a_dataset_len()
        assert len(test_b_paths) == test_dataset.get_b_dataset_len()

        for elem_idx, (a_img, b_img) in enumerate(test_dataset):
            print('Checking element [%d / %d]' % (elem_idx+1, len(test_dataset)), end='\r')
            assert self.test_image_shape == a_img.shape
            assert self.test_image_shape == b_img.shape
