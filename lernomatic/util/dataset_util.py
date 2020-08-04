"""
DATASET_UTIL
Just some dataset stuff

Stefan Wong 2019
"""

import torchvision


def get_mnist_datasets(data_dir:str, download:bool=True) -> tuple:
    dataset_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize( (0.1307,), (0.3081,))
    ])

    train_dataset = torchvision.datasets.MNIST(
        data_dir,
        train = True,
        download = download,
        transform = dataset_transform
    )
    val_dataset = torchvision.datasets.MNIST(
        data_dir,
        train = False,
        download = download,
        transform = dataset_transform
    )

    return (train_dataset, val_dataset)


def get_cifar10_datasets(data_dir:str, download:bool=True) -> tuple:
    dataset_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize( (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # training data
    train_dataset = torchvision.datasets.CIFAR10(
        data_dir,
        train = True,
        download = download,
        transform = dataset_transform
    )

    val_dataset = torchvision.datasets.CIFAR10(
        data_dir,
        train = False,
        download = download,
        transform = dataset_transform
    )

    return (train_dataset, val_dataset)
