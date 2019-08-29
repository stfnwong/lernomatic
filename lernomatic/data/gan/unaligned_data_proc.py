"""
UNALIGNED_DATA_PROC
Data processor for unaligned Cycle-GAN data

Stefan Wong 2019
"""

import os
import PIL
import h5py
import numpy as np
from tqdm import tqdm
from lernomatic.util import image_util



class UnalignedImageProc(object):
    """
    UnalignedImageProc
    Processs unaligned image data into and HDF5 file.
    For use with CycleGAN models

    Arguments:

        verbose: (bool)
            Set verbose mode

        image_dataset_name: (str)
            Name of image dataset in output HDF5 file (default: 'images')

        id_dataset_name: (str)
            Name of id dataset in output HDF5 file (default: 'ids')

        image_shape: (tuple)
            Shape of a single image in the dataset (default: (3, 256, 256))

    """
    def __init__(self, **kwargs) -> None:
        valid_directions = ('AB', 'BA')
        self.verbose:bool         = kwargs.pop('verbose', False)
        self.a_img_name:str       = kwargs.pop('a_img_name', 'a_imgs')
        self.b_img_name:str       = kwargs.pop('b_img_name', 'b_imgs')
        self.a_id_name:str        = kwargs.pop('a_id_name', 'a_ids')
        self.b_id_name:str        = kwargs.pop('b_id_name', 'b_ids')
        # image dimensions
        self.image_shape:tuple    = kwargs.pop('image_shape', (3, 256, 256))
        self.id_dataset_size:int  = kwargs.pop('id_dataset_size', 1)

    def __repr__(self) -> str:
        return 'UnalignedImageProc'

    def proc(self, a_paths:list, b_paths:list, outfile:str) -> None:
        # Note that for the unaligned dataset the paths don't need to be the
        # same length
        with h5py.File(outfile, 'w') as fp:
            a_images = fp.create_dataset(
                self.a_img_name,
                (len(a_paths),) + self.image_shape,
                dtype=np.uint8
            )
            b_images = fp.create_dataset(
                self.b_img_name,
                (len(b_paths),) + self.image_shape,
                dtype=np.uint8
            )
            a_ids = fp.create_dataset(
                self.a_id_name,
                (len(a_paths), self.id_dataset_size),
                dtype='S10'
            )
            b_ids = fp.create_dataset(
                self.b_id_name,
                (len(b_paths), self.id_dataset_size),
                dtype='S10'
            )

            invalid_file_list = []
            # Process the a_paths first
            print('Processing %d A paths' % len(a_paths))
            for idx, path in enumerate(tqdm(a_paths, total=len(a_paths), unit='files')):
                if not os.path.isfile(path):
                    invalid_file_list.append(path)
                    continue

                img_a = PIL.Image.open(path).convert('RGB')
                img_a = image_util.crop(img_a, 0, 0, self.image_shape[-1])
                img_a = np.asarray(img_a).transpose(2, 0, 1)

                a_images[idx] = img_a
                a_id, _     = os.path.splitext(os.path.basename(path))
                a_id        = a_id.encode('ascii')
                a_ids[idx]  = a_id

            # Then process the b paths
            print('Processing %d B paths' % len(b_paths))
            for idx, path in enumerate(tqdm(b_paths, total=len(b_paths), unit='files')):
                if not os.path.isfile(path):
                    invalid_file_list.append(path)
                    continue

                img_b = PIL.Image.open(path).convert('RGB')
                img_b = image_util.crop(img_b, 0, 0, self.image_shape[-1])
                img_b = np.asarray(img_b).transpose(2, 0, 1)
                # add image to dataset
                b_images[idx] = img_b
                b_id, _ = os.path.splitext(os.path.basename(path))
                b_id    = b_id.encode('ascii')
                b_ids[idx] = b_id

        # Print stats?
        print('Processed %d images, %s images failed' % (len(a_paths), len(invalid_file_list)))
