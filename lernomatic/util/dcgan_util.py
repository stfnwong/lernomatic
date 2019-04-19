"""
DCGAN_UTIL
Various functions for use with DCGAN

Stefan Wong 2019
"""

import torch
import torchvision.utils as vutils

def dcgan_gen_image_grid(gen_model, **kwargs):

    fixed_noise = kwargs.pop('fixed_noise', None)
    nz          = kwargs.pop('nz', 100)
    img_size    = kwargs.pop('img_size', 64)
    num_images  = kwargs.pop('num_images', 64)

    if fixed_noise is None:
        fixed_noise = torch.randn(img_size, nz, 1, 1)

    img_list = []
    gen_model.eval()

    for n in range(num_images):
        fake_g = gen_model(fixed_noise).detach().cpu()
        img_list.append(vutils.make_grid(fake_g, padding=2, normalize=True))

    return img_list
