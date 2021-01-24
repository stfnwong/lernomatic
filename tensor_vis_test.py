# Try visualising some tensors in alexnet

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models, utils
import argparse


def vis_tensor(tensor:torch.Tensor, ch:int=0, all_kernels:bool=False, num_rows:int=8, padding:int=1) -> None:
    N, C, W, H = tensor.shape

    if all_kernels:
        tensor = tensor.view(N * C, -1, W, H)
    elif C != 3:        # we sort of presume that we are looking at images
        tensor = tensor[:, ch, :, :].unsqueeze(dim=1)

    rows = np.min((tensor.shape[0] // num_rows + 1, 64))
    # prep the output image
    grid = utils.make_grid(tensor, nrow=num_rows, padding=padding, normalize=True)
    plt.figure(figsize=(num_rows, num_rows))
    plt.imshow(grid.numpy().transpose((1, 2, 0)))


if __name__ == '__main__':
    #from pudb import set_trace; set_trace()
    # conv layers in alexnet are 0, 3, 6, 8, 10
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--layer',
                        type=int,
                        default = 6,
                            help='Which layer to visualize'
                        )
    args = arg_parser.parse_args()

    # model stuff
    model = models.alexnet(pretrained=True)
    filter_tensor = model.features[args.layer].weight.data.clone()

    vis_tensor(filter_tensor, ch=0, all_kernels=False)

    plt.axis('off')
    plt.ioff()
    plt.show()
