"""
ALIGNED_DATA_PROC
Processors for an aligned dataset (eg: one for CycleGAN)

Stefan Wong 2019
"""

import h5py
import PIL
from PIL import Image       # TODO : replace with opencv at some point
import os
import numpy as np
from tqdm import tqdm
from lernomatic.data.gan import aligned_data_split
from lernomatic.util import image_util

# debug
from pudb import set_trace; set_trace()


class AlignedImageProc(object):
    """
    AlignedImageProc
    Base class for processing Aligned Image datasets, eg: for pix2pix models

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
        self.int_to_dir = {0 : 'AB', 1 : 'BA'}

        if self.direction not in self.dir_to_int:
            raise ValueError('Invalid direction [%s], must be one of %s' %\
                        (str(self.direction), str(valid_directions))
            )

    def __repr__(self) -> str:
        return 'AlignedImageProc'

    def get_dir_str(self) -> str:
        return self.direction

    def get_dir_int(self) -> int:
        return self.dir_to_int[self.direction]

    def align_image_arrays(self, a_img:np.ndarray, b_img:np.ndarray) -> np.ndarray:
        return np.concatenate([a_img, b_img], dim=1)

    #def check_and_resize(self, img:np.ndarray) -> tuple:
    def check_and_resize(self, img:PIL.Image) -> tuple:
        status_ok = True
        if len(img.size) == 2:
            aw, ah = img.size
        elif len(img.size) == 3:
            _, aw, ah = img
        #if len(img.shape) == 2:
        #    aw, ah = img.shape
        #elif len(img.shape) == 3:
        #    _, aw, ah = img.shape
        else:
            if self.verbose:
                print('Image [%s] must have shape (W, H) or shape (C, W, H)' % str(path))
            status_ok = False
        # Crop image to new size
        img = image_util.crop(img, 0, 0, self.image_shape[-1])

        return (img, status_ok)

    def proc(self) -> None:
        raise NotImplementedError('This method should be implemented in sub-class')



class AlignedImageJoin(AlignedImageProc):
    """
    AlignedImageJoin

    Take a set of A images and another set of B images and join them into a
    set of AB images.
    """
    def __init__(self, **kwargs) -> None:
        super(AlignedImageJoin, self).__init__(**kwargs)

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
        return 'AlignedImageJoin'

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

                # check that files exist, and if not, place into
                # invalid_file_list
                if (not os.path.isfile(a_img_path)) or (not os.path.isfile(b_img_path)):
                    invalid_file_list.append((a_img_path, b_img_path))
                    continue

                img_a = Image.open(a_img_path).convert('RGB')
                img_b = Image.open(b_img_path).convert('RGB')
                # convert to arrays
                img_a = np.asarray(img_a)
                img_b = np.asarray(img_b)
                # swap channels
                img_a = img_a.transpose(2, 0, 1)
                img_b = img_b.transpose(2, 0, 1)
                img_a, img_a_status = self.check_and_resize(img_a)
                img_b, img_b_status = self.check_and_resize(img_b)

                if (img_a_status is False) or (img_b_status is False):
                    invalid_file_list.append((a_img_path, b_img_path))
                    continue

                # concat
                if self.direction == 'AB':
                    aligned_image = np.concatenate([img_a, img_b], 1)
                else:
                    aligned_image = np.concatenate([img_b, img_a], 1)

                # add image to dataset
                images[idx] = aligned_image
                a_id, _     = os.path.splitext(os.path.basename(a_img_path))
                b_id, _     = os.path.splitext(os.path.basename(b_img_path))
                a_id        = a_id.encode('ascii')
                b_id        = b_id.encode('ascii')
                a_ids[idx]  = a_id
                b_ids[idx]  = b_id

        # Print stats?
        print('Processed %d images, %s images failed' % (len(a_paths), len(invalid_file_list)))



class AlignedImageSplit(AlignedImageProc):
    """
    AlignedImageSplit

    Take a set of AB images and split them into A and B images.
    """
    def __init__(self, **kwargs) -> None:
        self.a_img_name:str = kwargs.pop('a_img_name', 'a_imgs')
        self.b_img_name:str = kwargs.pop('b_img_name', 'b_imgs')
        super(AlignedImageSplit, self).__init__(**kwargs)

    def __repr__(self) -> str:
        return 'AlignedImageSplit'

    def proc(self, paths:list, outfile:str) -> None:
        with h5py.File(outfile, 'w') as fp:
            # For this kind of dataset, the output image is two images that are
            # concatenated together horizontally
            a_imgs = fp.create_dataset(
                self.a_img_name,
                (len(paths),) + self.image_shape,
                dtype=np.uint8
            )
            b_imgs = fp.create_dataset(
                self.b_img_name,
                (len(paths),) + self.image_shape,
                dtype=np.uint8
            )
            # Ids for each of the files
            fids = fp.create_dataset(
                'ids_' + str(self.id_dataset_name),
                (len(paths), self.id_dataset_size),
                dtype='S10'
            )

            invalid_file_list = []
            for idx, (path) in enumerate(tqdm(paths, total=len(paths), unit='paths')):
                # check that files exist, and if not, place into
                # invalid_file_list
                if not os.path.isfile(path):
                    invalid_file_list.append(path)
                    continue

                image = Image.open(path).convert('RGB')
                w, h = image.size
                # TODO : make crop direction settable?
                img_a = image.crop((0, 0, (w // 2), h))
                img_b = image.crop(((w // 2), 0, w, h))

                # Convert to numpy arrays
                #img_a, img_a_status = self.check_and_resize(img_a)
                #img_b, img_b_status = self.check_and_resize(img_b)
                img_a = image_util.crop(img_a, 0, 0, self.image_shape[-1])
                img_b = image_util.crop(img_b, 0, 0, self.image_shape[-1])

                img_a = np.asarray(img_a)
                img_a = img_a.transpose(2, 0, 1)
                img_b = np.asarray(img_b)
                img_b = img_b.transpose(2, 0, 1)

                # add image to dataset
                a_imgs[idx] = img_a
                b_imgs[idx] = img_b
                img_id, _   = os.path.splitext(os.path.basename(path))
                fids[idx]   = img_id.encode('ascii')

                #a_id, _     = os.path.splitext(os.path.basename(a_img_path))
                #b_id, _     = os.path.splitext(os.path.basename(b_img_path))
                #a_id        = a_id.encode('ascii')
                #b_id        = b_id.encode('ascii')

        # Print stats?
        print('Processed %d images, %s images failed' % (len(paths), len(invalid_file_list)))
