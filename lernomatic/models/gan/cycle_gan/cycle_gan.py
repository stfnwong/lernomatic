"""
CYCLE_GAN
Models for CycleGAN

Stefan Wong 2019
"""

import torch
import random
import torch.nn as nn


class ImagePool():
    """
    Implements an image buffer that stores previously generated images.
    """
    def __init__(self, pool_size) -> None:
        self.pool_size = pool_size
        if self.pool_size > 0:      # create an empty pool
            self.num_imgs = 0
            self.images = []

    def query(self, images) -> list:
        """
        QUERY
        Return an image from the pool

        Arguments:
            images     - latest generated images from the generator

        Output:
            out_images - set of images from the buffer. 50/100 odds that
                         the buffer will return input images. 50/100 odds
                         that the buffer will return images previously
                         stored and insert the current image into the buffer
        """

        # do nothing if the buffer size is 0
        if self.pool_size == 0:
            return images

        out_images = []
        for img in images:
            img = torch.unsqueeze(img.data, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs += 1
                self.images.append(img)
                out_images.append(img)
            else:
                p = random.uniform(0, 1)
                # by chance (50%) the buffer may return a previously stored image and
                # insert the current image into the buffer
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)   # note, randint is inclusive
                    tmp = self.images[random_id].clone()
                    out_images.append(tmp)
                # by another 50% chance the buffer will return the current
                # image
                else:
                    out_images.append(img)
        out_images = torch.cat(out_images, 0)
        return out_images
