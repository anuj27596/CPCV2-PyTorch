# Based On:
# https://github.com/loeweX/Greedy_InfoMax/blob/master/GreedyInfoMax/vision/data/get_dataloader.py

import torch
import torchvision
import torchvision.transforms as transforms
from data.image_preprocessing import patchify, patchify_augment

import os
from os import path
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

num_workers = 1

aug = {
    "stl10": {
        "randcrop": 64,
        "randcrop_padding": None,
        "rand_horizontal_flip": True,
        "grayscale": True,
        # values for train+unsupervised combined
        "mean": [0.44087532, 0.42790526, 0.3867924],
        "std": [0.26826888, 0.2610458, 0.2686684],
        "bw_mean":  [0.42709708],
        "bw_std": [0.257203],
    },
    "cifar10": {
        "randcrop": 32,
        "randcrop_padding": 4,
        "rand_horizontal_flip": True,
        "grayscale": True,
        "mean": [0.49139968, 0.48215827, 0.44653124],
        "std": [0.24703233, 0.24348505, 0.26158768],
        "bw_mean": [0.4808616],
        "bw_std": [0.23919088],
    },
    "cifar100": {
        "randcrop": 32,
        "randcrop_padding": 4,
        "rand_horizontal_flip": True,
        "grayscale": True,
        "mean": [0.5070746, 0.48654896, 0.44091788],
        "std": [0.26733422, 0.25643846, 0.27615058],
        "bw_mean": [0.48748648],
        "bw_std": [0.25063065],
    }
}

# Get transformations that are applied to "entire" image
def get_transforms(args, eval, aug):
    trans = []

    if aug["randcrop"] and not eval:
        trans.append(transforms.RandomCrop(aug["randcrop"], aug["randcrop_padding"]))

    if aug["randcrop"] and eval:
        trans.append(transforms.CenterCrop(aug["randcrop"]))

    if args.image_resize:   # User Input - Note this is after cropping
        trans.append(transforms.Resize(args.image_resize))

    if aug["rand_horizontal_flip"] and not eval:
        trans.append(transforms.RandomHorizontalFlip())

    if aug["grayscale"]:
        trans.append(transforms.Grayscale())
        trans.append(transforms.ToTensor())
        #trans.append(transforms.Normalize(mean=aug["bw_mean"], std=aug["bw_std"])) # I THINK THIS IS REMOVED IN CPCV2?
    elif aug["mean"]:
        trans.append(transforms.ToTensor())
        #trans.append(transforms.Normalize(mean=aug["mean"], std=aug["std"])) # I THINK THIS IS REMOVED IN CPCV2?
    else:
        trans.append(transforms.ToTensor())
    
    # Always patchify, during training if specified also augment
    if eval or not args.patch_aug:
        trans.append(patchify(grid_size=args.grid_size))
    else:
        trans.append(patchify_augment(grid_size=args.grid_size))

    trans = transforms.Compose(trans)

    return trans


def get_stl10_dataloader(args, labeled=False, validate=False):
    data_path = os.path.join("data", "stl10")

    # Define Transforms
    transform_train = transforms.Compose(
        [get_transforms(args, eval=False, aug=aug["stl10"])])
    transform_valid = transforms.Compose(
        [get_transforms(args, eval=True, aug=aug["stl10"])])

    # Get Datasets
    unsupervised_dataset = torchvision.datasets.STL10(
        data_path, split="unlabeled", transform=transform_train, download=args.download_dataset
    )
    train_dataset = torchvision.datasets.STL10(
        data_path, split="train", transform=transform_train, download=args.download_dataset
    )
    test_dataset = torchvision.datasets.STL10(
        data_path, split="test", transform=transform_valid, download=args.download_dataset
    )

    # Get DataLoaders
    unsupervised_loader = torch.utils.data.DataLoader(
        unsupervised_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers
    )

    # Take subset of training data for train_classifier
    try:
        train_size = args.train_size
        indices = list(range(len(train_dataset)))
        np.random.shuffle(indices)
        train_indices = indices[:train_size]
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=num_workers,
        )
    except AttributeError:  
        # args.train_size is not defined during train_CPC
        # train_loader is not needed during train_CPC
        train_loader = None

    return (unsupervised_loader, train_loader, test_loader)


def get_cifar_dataloader(args, cifar_classes):

    if cifar_classes == 10:
        data_path = os.path.join("data", "cifar10")

        # Define Transforms
        transform_train = transforms.Compose(
            [get_transforms(args, eval=False, aug=aug["cifar10"])])
        transform_valid = transforms.Compose(
            [get_transforms(args, eval=True, aug=aug["cifar10"])])

        # Get Datasets
        unsupervised_dataset = torchvision.datasets.CIFAR10(
            data_path, train=True, transform=transform_train, download=args.download_dataset
        )
        test_dataset = torchvision.datasets.CIFAR10(
            data_path, train=False, transform=transform_valid, download=args.download_dataset
        )

    if cifar_classes == 100:
        data_path = os.path.join("data", "cifar100")

        # Define Transforms
        transform_train = transforms.Compose(
            [get_transforms(args, eval=False, aug=aug["cifar100"])])
        transform_valid = transforms.Compose(
            [get_transforms(args, eval=True, aug=aug["cifar100"])])

        # Get Datasets
        unsupervised_dataset = torchvision.datasets.CIFAR100(
            data_path, train=True, transform=transform_train, download=args.download_dataset
        )
        test_dataset = torchvision.datasets.CIFAR100(
            data_path, train=False, transform=transform_valid, download=args.download_dataset
        )

    # Get DataLoaders
    unsupervised_loader = torch.utils.data.DataLoader(
        unsupervised_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers
    )

    # Take subset of training data for train_classifier
    try:
        train_size = args.train_size
        indices = list(range(len(unsupervised_dataset)))
        np.random.shuffle(indices)
        train_indices = indices[:train_size]
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
        train_loader = torch.utils.data.DataLoader(
            unsupervised_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=num_workers,
        )
    except AttributeError:  
        # args.train_size is not defined during train_CPC
        # train_loader is not needed during train_CPC
        train_loader = None

    return (unsupervised_loader, train_loader, test_loader)


def get_cifar10_dataloader(args):
    unsupervised_loader, train_loader, test_loader = get_cifar_dataloader(
        args, 10)
    return (unsupervised_loader, train_loader, test_loader)


def get_cifar100_dataloader(args):
    unsupervised_loader, train_loader, test_loader = get_cifar_dataloader(
        args, 100)
    return (unsupervised_loader, train_loader, test_loader)


def create_validation_sampler(dataset_size):
    # Creating data indices for training and validation splits:
    validation_split = 0.2
    shuffle_dataset = True

    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating data samplers and loaders:
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)

    return train_sampler, valid_sampler


def calculate_normalisation(dataset):
    if dataset == "stl10":
        data_path = os.path.join("data", "stl10")

        # RGB
        train_transform = transforms.Compose([transforms.ToTensor()])
        train_set = torchvision.datasets.STL10(
            root=data_path, split="train+unlabeled", download=True, transform=train_transform)

        c1 = np.concatenate([np.asarray(train_set[i][0][0]) for i in range(
            len(train_set))])  # concatenate each channel
        c2 = np.concatenate([np.asarray(train_set[i][0][1])
                             for i in range(len(train_set))])
        c3 = np.concatenate([np.asarray(train_set[i][0][2])
                             for i in range(len(train_set))])

        train_mean = [np.mean(c1, axis=(0, 1,)), np.mean(
            c2, axis=(0, 1,)), np.mean(c3, axis=(0, 1,))]
        train_std = [np.std(c1, axis=(0, 1)), np.std(
            c2, axis=(0, 1)), np.std(c3, axis=(0, 1))]

        # grayscale
        train_transform = transforms.Compose(
            [transforms.Grayscale(), transforms.ToTensor()])
        train_set = torchvision.datasets.STL10(
            root=data_path, split="train+unlabeled", download=True, transform=train_transform)

        c = np.concatenate([np.asarray(train_set[i][0][0])
                            for i in range(len(train_set))])

        grey_train_mean = [np.mean(c, axis=(0, 1,))]
        grey_train_std = [np.std(c, axis=(0, 1))]

        # [0.44087532, 0.42790526, 0.3867924] [0.26826888, 0.2610458, 0.2686684]
        # [0.42709708] [0.257203]

    elif dataset == "cifar10":
        data_path = os.path.join("data", "cifar10")

        # RGB
        train_transform = transforms.Compose([transforms.ToTensor()])
        train_set = torchvision.datasets.CIFAR10(
            root=data_path, train=True, download=True, transform=train_transform)

        c1 = np.concatenate([np.asarray(train_set[i][0][0])
                             for i in range(len(train_set))])
        c2 = np.concatenate([np.asarray(train_set[i][0][1])
                             for i in range(len(train_set))])
        c3 = np.concatenate([np.asarray(train_set[i][0][2])
                             for i in range(len(train_set))])

        train_mean = [np.mean(c1, axis=(0, 1,)), np.mean(
            c2, axis=(0, 1,)), np.mean(c3, axis=(0, 1,))]
        train_std = [np.std(c1, axis=(0, 1)), np.std(
            c2, axis=(0, 1)), np.std(c3, axis=(0, 1))]

        # grayscale
        train_transform = transforms.Compose(
            [transforms.Grayscale(), transforms.ToTensor()])
        train_set = torchvision.datasets.CIFAR10(
            root=data_path, train=True, download=True, transform=train_transform)

        c = np.concatenate([np.asarray(train_set[i][0][0])
                            for i in range(len(train_set))])

        grey_train_mean = [np.mean(c, axis=(0, 1,))]
        grey_train_std = [np.std(c, axis=(0, 1))]

        # [0.49139968, 0.48215827, 0.44653124] [0.24703233, 0.24348505, 0.26158768]
        # [0.4808616] [0.23919088]

    elif dataset == "cifar100":
        data_path = os.path.join("data", "cifar100")

        # RGB
        train_transform = transforms.Compose([transforms.ToTensor()])
        train_set = torchvision.datasets.CIFAR100(
            root=data_path, train=True, download=True, transform=train_transform)

        c1 = np.concatenate([np.asarray(train_set[i][0][0])
                             for i in range(len(train_set))])
        c2 = np.concatenate([np.asarray(train_set[i][0][1])
                             for i in range(len(train_set))])
        c3 = np.concatenate([np.asarray(train_set[i][0][2])
                             for i in range(len(train_set))])

        train_mean = [np.mean(c1, axis=(0, 1,)), np.mean(
            c2, axis=(0, 1,)), np.mean(c3, axis=(0, 1,))]
        train_std = [np.std(c1, axis=(0, 1)), np.std(
            c2, axis=(0, 1)), np.std(c3, axis=(0, 1))]

        # grayscale
        train_transform = transforms.Compose(
            [transforms.Grayscale(), transforms.ToTensor()])
        train_set = torchvision.datasets.CIFAR100(
            root=data_path, train=True, download=True, transform=train_transform)

        c = np.concatenate([np.asarray(train_set[i][0][0])
                            for i in range(len(train_set))])

        grey_train_mean = [np.mean(c, axis=(0, 1,))]
        grey_train_std = [np.std(c, axis=(0, 1))]

        # [0.5070746, 0.48654896, 0.44091788] [0.26733422, 0.25643846, 0.27615058]
        # [0.48748648] [0.25063065]

    print(train_mean, train_std)
    print(grey_train_mean, grey_train_std)


if __name__ == "__main__":
    calculate_normalisation("stl10")
    calculate_normalisation("cifar10")
    calculate_normalisation("cifar100")
