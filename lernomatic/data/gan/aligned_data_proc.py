"""
ALIGNED_DATA_PROC
Processors for an aligned dataset (eg: one for CycleGAN)

Stefan Wong 2019
"""

import h5py
import cv2
import os
import numpy as np
from tqdm import tqdm
from lernomatic.data.gan import aligned_data_split
from lernomatic.util import image_util

# debug
#from pudb import set_trace; set_trace()


def align_image_arrays(a_img:np.ndarray,
                            b_img:np.ndarray) -> np.ndarray:
    return np.concatenate([a_img, b_img], dim=1)


class AlignedImageProc(object):
    """
    ALIGINEDIMAGEPROC
    Processs aligned image data into and HDF5 file.
    For use with pix2pix models

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
        self.verbose:bool               = kwargs.pop('verbose', False)
        self.image_dataset_name:str     = kwargs.pop('image_dataset_name', 'images')
        self.id_dataset_name:str        = kwargs.pop('id_dataset_name', 'ids')

        # image dimensions
        self.image_shape:tuple          = kwargs.pop('image_shape', (3, 256, 256))
        self.id_dataset_size:int        = kwargs.pop('id_dataset_size', 1)
        # which direction?
        self.direction_dataset_name:str = kwargs.pop('direction_dataset_name', 'direction')
        self.direction_dataset_size:int = kwargs.pop('direction_dataset_size', 1)
        self.direction:str              = kwargs.pop('direction', 'AB')

        self.dir_to_int = {'AB' : 0, 'BA' : 1}
        self.int_to_dir = {0 : 'Ab', 1 : 'BA'}

        if self.direction not in self.dir_to_int:
            raise ValueError('Invalid direction [%s], must be one of %s' %\
                        (str(self.direction), str(valid_directions))
            )

        # the output image will actually be concatenated along dim=1 (width)
        if len(self.image_shape) != 3:
            raise ValueError('image_shape tuple must have 3 dimensions')

        out_img_shape = []
        for n, dim in enumerate(self.image_shape):
            if n == 1:
                out_img_shape.append(2 * dim)
            else:
                out_img_shape.append(dim)
        self.out_img_shape = tuple(out_img_shape)

    def __repr__(self) -> str:
        return 'AlignedImageProc'

    def check_and_crop(self, img_a:np.ndarray, img_b:np.ndarray) -> tuple:
        status_ok = True

        for img in [img_a, img_b]:
            if len(img.shape) == 2:
                aw, ah = img.shape
            elif len(img.shape) == 3:
                _, aw, ah = img.shape
            else:
                if self.verbose:
                    print('Image [%s] must have shape (W, H) or shape (C, W, H)' % str(path))
                status_ok = False
                break

        # Crop image to new size
        img_a = image_util.crop(img_a, 0, 0, self.image_shape[-1])
        img_b = image_util.crop(img_b, 0, 0, self.image_shape[-1])

        return (img_a, img_b, status_ok)

    def proc(self, a_paths:list, b_paths:list, outfile:str) -> None:
        if len(a_paths) != len(b_paths):
            raise ValueError('Number of a_paths (%d) must match number of b_paths (%d)' %\
                        (len(a_paths), len(b_paths)),
            )

        with h5py.File(outfile, 'w') as fp:
            # For this kind of dataset, the output image is two images that are
            # concatenated together horizontally
            images = fp.create_dataset(
                self.image_dataset_name,
                (len(a_paths),) + self.out_img_shape,
                dtype=np.uint8
            )
            a_ids = fp.create_dataset(
                'B_' + str(self.id_dataset_name),
                (len(a_paths), self.id_dataset_size),
                dtype='S10'
            )
            b_ids = fp.create_dataset(
                'A_' + str(self.id_dataset_name),
                (len(a_paths), self.id_dataset_size),
                dtype='S10'
            )

            # add attributes
            images.attrs['direction'] = self.direction

            invalid_file_list = []
            for idx, (a_img_path, b_img_path) in enumerate(
                tqdm(zip(a_paths, b_paths), total=len(a_paths), unit='paths')):
                # read images

                # check that files exist, and if not, place into
                # invalid_file_list
                if (not os.path.isfile(a_img_path)) or (not os.path.isfile(b_img_path)):
                    invalid_file_list.append((a_img_path, b_img_path))
                    continue

                img_a = cv2.imread(a_img_path)
                img_b = cv2.imread(b_img_path)
                img_a = img_a.transpose(2, 0, 1)
                img_b = img_b.transpose(2, 0, 1)
                img_a, img_b, status_ok = self.check_and_crop(img_a, img_b)

                if status_ok is False:
                    invalid_file_list.append((a_img_path, b_img_path))
                    continue

                # concat
                if self.direction == 'AB':
                    aligned_image = np.concatenate([img_a, img_b], 1)
                else:
                    aligned_image = np.concatenate([img_b, img_a], 1)

                # add image to dataset
                images[idx] = aligned_image
                a_id, _ = os.path.splitext(os.path.basename(a_img_path))
                b_id, _ = os.path.splitext(os.path.basename(b_img_path))
                a_id    = a_id.encode('ascii')
                b_id    = b_id.encode('ascii')
                a_ids[idx] = a_id
                b_ids[idx] = b_id

        # Print stats?
        print('Processed %d images, %s images failed' % (len(a_paths), len(invalid_file_list)))
