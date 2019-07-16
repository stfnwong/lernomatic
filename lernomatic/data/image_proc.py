"""
IMAGE_PROC
Data processing for image datasets

Stefan Wong 2019
"""

import h5py
import cv2
import numpy as np
from tqdm import tqdm
from lernomatic.data import data_split as lm_data_split

# debug
#from pudb import set_trace; set_trace()

class ImageDataProc(object):
    """
    ImageDataProc
    Process a generic image dataset

    Arguments:
        image_dataset_name: (str)
            Name of image dataset in output HDF5 file (default: 'images')

        image_dataset_size: (tuple)
            Shape of a single image in the image dataset (default: (3, 224, 224))

        label_dataset_name: (str)
            Name of label dataset in output HDF5 file (default: 'labels')

        label_dataset_size: (tuple)
            Shape of a single label in the image dataset (default: 1)

        label_dataset_dtype: (int)
            Datatype of the labels in the label dataset

        id_dataset_name: (str)
            Name of id dataset in output HDF5 file (default: 'ids')

        id_dtype:
            Datatype of the ids in the id dataset

        verbose: (bool)
            Set verbose mode

    """
    def __init__(self, **kwargs) -> None:
        self.verbose = kwargs.pop('verbose', False)
        # dataset options
        self.image_dataset_name  : str   = kwargs.pop('image_dataset_name', 'images')
        self.image_dataset_size  : tuple = kwargs.pop('image_dataset_size', (3, 224, 224))
        self.label_dataset_name  : str   = kwargs.pop('label_dataset_name', 'labels')
        self.label_dataset_size  : int   = kwargs.pop('label_dataset_size', 1)
        self.label_dataset_dtype         = kwargs.pop('label_dataset_dtype', int)
        self.id_dataset_name     : str   = kwargs.pop('id_dataset_name', 'ids')
        self.id_dtype                    = kwargs.pop('id_dtype', int)

    def __repr__(self) -> str:
        return 'ImageDataProc'

    def __len__(self) -> int:
        return self.dataset_size

    def proc(self, data_split:lm_data_split.DataSplit, outfile:str) -> None:
        """
        proc()
        Process a split into an HDF5 file
        """
        with h5py.File(outfile, 'w') as fp:
            images = fp.create_dataset(
                self.image_dataset_name,
                (len(data_split),) + self.image_dataset_size,
                dtype=np.uint8
            )
            ids = fp.create_dataset(
                self.id_dataset_name,
                (len(data_split), self.label_dataset_size),
                dtype=int
            )
            labels = fp.create_dataset(
                self.label_dataset_name,
                (len(data_split), self.label_dataset_size),
                dtype=int
            )

            invalid_file_list = []
            for n, (img_path, img_id, label) in enumerate(tqdm(data_split, unit='images')):
                img = cv2.imread(img_path)

                if img is None:
                    invalid_file_list.append(img_path)
                    images[n] = np.zeros(self.image_dataset_size)
                    ids[n]    = 0
                    labels[n] = -1
                    continue

                img = cv2.resize(img, self.image_dataset_size[1:], interpolation=cv2.INTER_CUBIC)
                if len(img.shape) != self.image_dataset_size[0]:
                    print('img channels (%d) != required (%d)' % (len(img.shape), self.image_dataset_size[0]))

                if img.shape[-1] == self.image_dataset_size[0]:
                    img = img.transpose(2,0,1)

                images[n] = img
                ids[n]    = img_id
                labels[n] = label

        if self.verbose:
            print('%d invalid files found out of %d total (%.3f %%)' %\
                  (len(invalid_file_list), len(data_split), 100 * (len(invalid_file_list) + 1e-8) / len(data_split))
            )
