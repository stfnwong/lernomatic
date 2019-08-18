"""
IMAGE_PROC
Data processing for image datasets

Stefan Wong 2019
"""

import h5py
import cv2   # Should be PIL for consistency, but cv2 is actually the better choice overall
from PIL import Image
import numpy as np
from tqdm import tqdm
from lernomatic.data import data_split
from lernomatic.util import image_util


# debug
from pudb import set_trace; set_trace()


class ImageDataProc(object):
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
        # format options
        self.to_pil              :bool   = kwargs.pop('to_pil', False)
        self.pil_rgb_format      :str    = kwargs.pop('pil_rgb_format', 'RGB')
        # TODO : to_tensor (ie: place a tensor directly into HDF5)

    def __repr__(self) -> str:
        return 'ImageDataProc'

    def __len__(self) -> int:
        return self.dataset_size

    def proc(self, split_data:data_split.DataSplit, outfile:str) -> None:
        with h5py.File(outfile, 'w') as fp:
            images = fp.create_dataset(
                self.image_dataset_name,
                (len(split_data),) + self.image_dataset_size,
                dtype=np.uint8
            )
            ids = fp.create_dataset(
                self.id_dataset_name,
                (len(split_data), self.label_dataset_size),
                dtype=int
            )
            labels = fp.create_dataset(
                self.label_dataset_name,
                (len(split_data), self.label_dataset_size),
                dtype=int
            )

            invalid_file_list = []
            for n, (img_path, img_id, label) in enumerate(tqdm(split_data, unit='images')):
                img = cv2.imread(img_path)

                if img is None:
                    invalid_file_list.append(img_path)
                    images[n] = np.zeros(self.image_dataset_size)
                    ids[n]    = 0
                    labels[n] = -1
                    continue

                if self.to_pil:
                    img = cv2.resize(img, self.image_dataset_size[0:2], interpolation=cv2.INTER_CUBIC)
                    img = Image.fromarray(img.astype('uint8'),  self.pil_rgb_format)
                else:
                    img = cv2.resize(img, self.image_dataset_size[1:], interpolation=cv2.INTER_CUBIC)
                    img = img.transpose(2, 0, 1)
                    # retain the image in ndarray format
                    if len(img.shape) != self.image_dataset_size[0]:
                        print('img channels (%d) != required (%d)' % (len(img.shape), self.image_dataset_size[0]))

                    if img.shape[-1] == self.image_dataset_size[0]:
                        img = img.transpose(2,0,1)

                images[n] = img
                ids[n]    = img_id
                labels[n] = label

        if self.verbose:
            print('%d invalid files found out of %d total (%.3f %%)' %\
                  (len(invalid_file_list), len(split_data), 100 * (len(invalid_file_list) + 1e-8) / len(split_data))
            )
