"""
IMAGE_POOL
Buffering for CycleGAN implementations

Heavily adapted from the implementation at (https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
"""

import random
import torch


class ImagePool(object):
    """
    Implements an image buffer that stores previously generated images, thus
    allowing us to update discriminators using a history of generated images
    """
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.imgs = []

    def query(self, in_images):
        """
        QUERY
        Return an image from the buffer. The image returned
        will be

        1) 50/50  - return input image
        2) 50/100 - return image previously stored in buffer
                    and insert current image into buffer.
        """

        if self.pool_size == 0:
            return in_images

        out_images = []
        for img in in_images:
            img = torch.unsqueeze(img.data, 0)
            # if the buffer isn't full, keep inserting
            if self.num_imgs < self.pool_size:
                self.imgs.append(img)
                self.num_imgs += 1
                out_images.append(img)
            else:
                p = random.uniform(0, 1)
                # flip a coin to see whether or not the buffer returns
                # the current image or a previously stored image
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size-1)
                    tmp = self.imgs[random_id].clone()
                    self.img[random_id] = img
                    out_images.append(tmp)
                else:
                    out_images.append(img)

        # collect images and return
        return torch.cat(out_images, 0)
