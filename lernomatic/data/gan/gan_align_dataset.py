"""
GAN_ALIGN_DATASET
Align data for CycleGAN stuff

Stefan Wong 2019
"""

import os
import cv2
import numpy as np


# TODO : make HDF5?
# TODO : write the inner loop part as a function?
# OR write so that it writes directly to HDF5, that way there isn't
# a big list sittng in memory


def align_images_from_array(a_img:np.ndarray,
                            b_img:np.ndarray) -> np.ndarray:
    return np.concatenate([a_img, b_img], dim=1)

def align_images_from_path(a_path:str,
                 b_path:str) -> np.ndarray:
    img_a = cv2.imread(a_path)
    img_b = cv2.imread(b_path)

    if img_a.shape != img_b.shape:
        raise ValueError('img_a.shape (%s) != img_b.shape (%s)' %\
                            (str(img_a.shape), str(img_b.shape))
        )
    aligned_img = np.concatenate([img_a, img_b], 1)

    return aligned_img


# TODO : factor above function into below one in next commit
def align_images_from_paths(a_data_paths:list,
                 b_data_paths:list,
                 verbose:bool=False) -> list:
    #if not os.path.exists(out_path):
    #    os.makedirs(out_path)

    if len(a_data_paths) != len(b_data_path):
        raise ValueError('len(a_data_paths) [%d] does not match len(b_data_paths) [%d]' %\
                    (len(a_data_paths), len(b_data_paths))
        )

    out_images = []
    for img_idx, (path_a, path_b) in enumerate(zip(a_data_paths, b_data_paths)):
        if verbose:
            print('Aligning image [%d / %d] ' % (img_idx+1, len(a_data_paths)), end='\r')

        img_a = cv2.imread(path_a)
        img_b = cv2.imread(path_b)

        if img_a.shape != img_b.shape:
            raise ValueError('img_a.shape (%s) != img_b.shape (%s)' %\
                             (str(img_a.shape), str(img_b.shape))
            )
        # images are aligned horizontally
        #aligned_img = np.zeros(img_a.shape[0] * 2, img_a.shape[1])
        #aligned_img[0 : img_a.shape[0], 0 : img_a.shape[1]] = img_a
        #aligned_img[img_a.shape[0] : 2 * img_a.shape[0], img_a.shape[1] : 2 * img_a.shape[1]] = img_b

        aligned_img = np.concatenate([img_a, img_b], 1)
        out_images.append(aligned_img)

    if verbose:
        print('\n DONE')

    return out_images
