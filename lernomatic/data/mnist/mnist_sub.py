"""
MNIST_SUB
Sub-sample parts of the MNIST dataset

Stefan Wong 2019
"""

from collections import namedtuple
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms


default_mnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    )]
)

class MNISTSub(torchvision.datasets.MNIST):
    def __init__(self, data_root:str, **kwargs) -> None:
        self.k:int            = kwargs.pop('k', 3000)
        self.train:bool       = kwargs.pop('train', True)
        self.download:bool    = kwargs.pop('download', True)
        self.transform        = kwargs.pop('transform', default_mnist_transform)
        self.target_transform = kwargs.pop('target_transform', None)

        super(MNISTSub, self).__init__(
            data_root,
            self.train,
            self.transform,
            self.target_transform,
            download=self.download
        )

    def __len__(self) -> int:
        if self.train:
            return self.k
        else:
            return 10000


def gen_mnist_subset(data_dir:str, num_elem:int=300, transform=None) -> tuple:

    if transform is None:
        transform = default_mnist_transform

    origin_train_data = torchvision.datasets.MNIST(
        data_dir,
        train=True,
        download=True,
        transform=transform
    )

    train_label_index = []
    valid_label_index = []
    for i in range(10):
        train_label_list = origin_train_data.train_labels.numpy()
        label_index = np.where(train_label_list == i)[0]
        label_subindex = list(label_index[0 : num_elem])
        valid_subindex = list(label_index[num_elem : 1000 + num_elem])
        train_label_index += label_subindex
        valid_label_index += valid_subindex


    # convert the tensors in the original dataset into arrays
    trainset_np = origin_train_data.train_data.numpy()
    trainset_label_np = origin_train_data.train_labels.numpy()
    train_data_sub = torch.from_numpy(trainset_np[train_label_index])


    trainset_new = MNISTSub(
        data_dir,
        train=True,
        download=True,
        transform=transform,
        k=3000
    )
    trainset_new.train_data = train_data_sub.clone()
    trainset_new.train_labels = train_labels_sub.clone()

    # original file dumps to disk here

    validset_np = origin_train_data.train_data.numpy()
    validset_label_np = origin_train_data.train_labels.numpy()
    valid_data_sub = torch.from_numpy(validset_np[valid_label_index])
    valid_labels_sub = torch.from_numpy(validset_label_np[valid_label_index])

    validset_new = MNISTSub(
        data_dir,
        train=False,
        download=True,
        transform=transform,
        k=10000
    )
    validset_new.test_data = valid_data_sub.clone()
    validset_new.test_labels = valid_data_labels_sub.clone()

    # dump() validation set here...
    train_unlabel_index = []
    for label_idx in range(60000):
        if label_idx in train_label_index or label_idx in valid_label_index:
            pass
        else:
            train_unlabel_index.append(label_idx)

    trainset_np = origin_train_data.train_data.numpy()
    trainset_label_np = orgin_train_data.train_labels.numpy()
    train_data_sub_unl = torch.from_numpy(trainset_np[train_unlabel_index])
    train_labels_sub_unl = torch.from_numpy(trainset_label_np[train_unlabel_index])

    # new unlabelled trainset
    trainset_unl_new = MNISTSub(
        data_dir,
        train=False,
        download=True,
        transform=transform,
        k=47000
    )
    trainset_unl_new.train_data = train_data_sub_unl.clone()
    trainset_unl_new.train_labels = None

    return (trainset_new, validset_new, trainset_unl_new)

