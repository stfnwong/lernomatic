# Test using a vanilla gradient to generate saliency maps

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse

from torchvision import models
from lernomatic.util.image_util import img_to_tensor
from lernomatic.vis.util import img_to_var


class VanillaBackprop:
    def __init__(self, model:nn.Module, target_layer:int=0) -> None:
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        # put model in eval mode
        self.model.eval()
        # set the layer hook for the target layer
        self._set_layer_hook(self.target_layer)

    def __repr__(self) -> str:
        return f"VanillaBackprop [{self.target_layer}]"

    def __str__(self) -> str:
        return self.__repr__()

    def _set_layer_hook(self, layer:int=0) -> None:
        def hook_fn(module: nn.Module, grad_in: torch.Tensor, grad_out: torch.Tensor) -> None:
            self.gradients = grad_in[0]

        # register hook to layer k
        target_layer = list(self.model._modules.items())[layer][1]
        target_layer.register_backward_hook(hook_fn)

    def get_grads(self, X:torch.Tensor, target_class:int) -> np.ndarray:
        # do forward pass
        model_output = self.model(X)
        self.model.zero_grad()
        # get target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # backward pass
        model_output.backward(gradient=one_hot_output)
        # convert to numpy array - size seems to small here...?
        # TODO: for Alexnet we expect the output to have shape (1, 3, 224, 224)
        grad_array = self.gradients.data.numpy()[0]

        return grad_array



if __name__ == '__main__':
    #from pudb import set_trace; set_trace()
    # conv layers in alexnet are 0, 3, 6, 8, 10
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('img',
                        type=str,
                        help='Path to test image file'
                        )
    arg_parser.add_argument('--target-layer',
                        type=int,
                        default = 0,
                        help='Which layer to visualize'
                        )
    arg_parser.add_argument('--target-class',
                        type=int,
                        default = 1,
                        help='Which class to visualize'
                        )
    args = arg_parser.parse_args()

    target_layer = args.target_layer
    target_example = args.target_class      # category: Snake
    img_path = args.img

    test_img = Image.open(img_path).convert("RGB")
    test_img_var = img_to_var(test_img)

    print(f"test_img_var.shape : {test_img_var.shape}")

    model = models.alexnet(pretrained=True)
    # get the grads
    grad_viz = VanillaBackprop(model, target_layer)
    grads = grad_viz.get_grads(test_img_var, target_example)

    # normalize grads before viz
    grads = grads - grads.min()
    grads /= grads.max()

    plt.imshow(grads)
    plt.axis('off')
    plt.ioff()
    plt.show()
