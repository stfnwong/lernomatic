# Try producing an image that minimizes the loss of a convolution
# operation for a given layer and filter of a CNN


import torch
from torch import nn
from torch.optim import Adam
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models, utils
from PIL import Image

from lernomatic.util.image_util import img_to_tensor
from lernomatic.vis.util import img_to_var



def vis_cnn_layer(model:nn.Module, target_layer:int, target_filter:int, **kwargs) -> None:
    img_w:int = kwargs.pop('img_w', 224)
    img_h:int = kwargs.pop('img_h', 224)
    max_iterations:int = kwargs.pop('max_iterations', 32)
    vis_when:int       = kwargs.pop('vis_when', 0)

    #from pudb import set_trace; set_trace()

    # TODO: replace with calls in util, update calls in util
    random_image = np.uint8(np.random.uniform(150, 180, (224, 224, 3)))
    image_var = img_to_var(random_image)

    optim = Adam([image_var], lr=0.1, weight_decay=1e-6)

    for i in range(1, max_iterations):
        optim.zero_grad()
        x = image_var
        for idx, layer in enumerate(model.features):
            # perform forward pass on each layer up to the target layer
            x = layer(x)
            if idx == target_layer:
                break
        # Now we have a tensor that contains the output from the
        # target layer. The shape of the tensor will be
        #
        # (1, Fl, Hl, Wl)
        #
        # Where Fl is the number of filters in layer l, and Hl and Wl
        # are the output height and width of layer l respectively.
        conv_output = x[0, target_filter]
        # now try to minimize the mean of the output for the target filter
        loss = -torch.mean(conv_output)
        print("[VIS_CNN_LAYER] Iteration [%d / %d], loss : %.3f conv_img_shape: %s " %\
              (i, max_iterations, loss.item(), str(conv_output.shape)))
        loss.backward()
        optim.step()

        #if (vis_when > 0)  and (i % vis_when == 0):
        #    vis_img = var_to_img(conv_output)

    plt.imshow(conv_output.detach().numpy())
    plt.axis('off')
    plt.ioff()
    plt.show()


if __name__ == '__main__':
    # conv layers in alexnet are 0, 3, 6, 8, 10
    layer = 6

    img_path = "/home/kreshnik/Pictures/shirley-head-2.png"
    test_img = Image.open(img_path).convert("RGB")
    test_img_var = img_to_var(test_img)

    model = models.alexnet(pretrained=True)
    filter_tensor = model.features[layer].weight.data.clone()

    target_layer = 17
    target_filter = 5

    vis_cnn_layer(model, target_layer, target_filter)
    #vis_cnn_layer(model, ch=0, all_kernels=False)

