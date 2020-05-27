"""
TEST_LMDB_DATASET
Unit tests for LMDB dataset object

Stefan Wong 2019
"""

import PIL
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision

# unit(s) under test
from lernomatic.data import lmdb_dataset


class TestLMDBDataset:
    test_lmdb_root = '/home/kreshnik/ml-data/dining_room_train_lmdb/'

    def test_init(self) -> None:
        dataset = lmdb_dataset.LMDBDataset(
            self.test_lmdb_root
        )

        print('dataset contains %d items' % len(dataset))

        for n, (image, target) in enumerate(dataset):
            print('Checking element [%d / %d]' % (n+1, len(dataset)), end='\r')
            assert isinstance(image, PIL.Image.Image)
            assert target == 0
            #print('Element %d : [%s] : %s' % (n, type(image), type(target)))
        print('\n OK')
