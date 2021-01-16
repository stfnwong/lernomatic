"""
VIZ UTILS
"""

import torch
from torch.autograd import Variable
import numpy as np
from PIL import Image

from typing import List, Tuple



def img_to_var(pil_img:Image, **kwargs) -> Variable:
    resize:bool      = kwargs.pop('resize', True)
    # default mean and std are from Imagenet dataset
    mean:List[float] = kwargs.pop("mean", [0.485, 0.456, 0.406])
    std:List[float]  = kwargs.pop("std", [0.229, 0.224, 0.225])
    # default output size is imagenet image size
    out_size:Tuple[int, int] = kwargs.pop('out_size', (224, 224))

    # try to convert other image types to PIL images
    if not isinstance(pil_img, Image.Image):
        try:
            pil_img = Image.fromarray(pil_img)
        except Exception as e:
            print("Failed to transform to PIL image object [%s]" % str(e))
            return None

    if resize:
        pil_img = pil_img.resize(out_size) #, Image.ANTIALIAS)

    # convert to torch variable
    im_array = np.float32(pil_img)
    im_array = im_array.transpose(2, 0, 1)      # flip to C, W, H

    for channel, _ in enumerate(im_array):
        im_array[channel] /= 255
        im_array[channel] -= mean[channel]
        im_array[channel] /= std[channel]

    # convert to N,C,W,H float tensor
    im_tensor = torch.from_numpy(im_array).float()
    im_tensor = im_tensor.unsqueeze(0)

    print('im_tensor.shape : %s' % str(im_tensor.shape))

    return Variable(im_tensor, requires_grad=True)


def var_to_img(im_var:Variable, **kwargs) -> Image:
    # default mean and std are inverse of mean, std from Imagenet dataset
    mean:List[float] = kwargs.pop("mean", [-0.485, -0.456, -0.406])
    std:List[float]  = kwargs.pop("std", [1/0.229, 1/0.224, 1/0.225])

    im_array = im_var.detach().numpy()[0]       # drop leading N dimension

    for channel, _ in enumerate(im_array):
        im_array[channel] -= mean[channel]
        im_array[channel] /= std[channel]

    # clamp
    im_array[im_array > 1] = 1
    im_array[im_array < 0] = 0
    im_array = np.round(im_array * 255)     # norm back to 8bpp
    im_array = np.uint8(im_array).transpose(1, 2, 0)

    return im_array
