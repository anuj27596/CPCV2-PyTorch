# Based On:
# https://github.com/loeweX/Greedy_InfoMax/blob/master/GreedyInfoMax/vision/data/get_dataloader.py

import torch
import torchvision
import torchvision.transforms as transforms
from data.image_preprocessing import *

from data.noisy_cifar import Noisy_CIFAR10, Noisy_CIFAR100

import medmnist

from data.cluster_dataset import BloodMNISTClusters

import os
import numpy as np

from tqdm import tqdm


aug = {
    "stl10": {
        # values for train+unsupervised combined
        "mean": [0.44087532, 0.42790526, 0.3867924],
        "std": [0.26826888, 0.2610458, 0.2686684],
        "bw_mean":  [0.42709708],
        "bw_std": [0.257203],
    },
    "cifar10": {
        "mean": [0.49139968, 0.48215827, 0.44653124],
        "std": [0.24703233, 0.24348505, 0.26158768],
        "bw_mean": [0.4808616],
        "bw_std": [0.23919088],
    },
    "cifar100": {
        "mean": [0.5070746, 0.48654896, 0.44091788],
        "std": [0.26733422, 0.25643846, 0.27615058],
        "bw_mean": [0.48748648],
        "bw_std": [0.25063065],
    },
    "breastmnist": {
        "mean": [0.3275805, 0.3275805, 0.3275805],
        "std": [0.20573014, 0.20573014, 0.20573014],
        "bw_mean": [0.3275805],
        "bw_std": [0.20573014],
    },
    "retinamnist": {
        "mean": [0.39838937, 0.24472676, 0.15575323],
        "std": [0.29828647, 0.20048243, 0.15065953],
        "bw_mean": [0.28052062],
        "bw_std": [0.22027038],
    },
    "bloodmnist": {
        "mean": [0.7943478, 0.659659, 0.6961932],
        "std": [0.21563025, 0.24160342, 0.11788896],
        "bw_mean": [0.7041403],
        "bw_std": [0.21754295],
    },
    "bloodclusters": {
        "mean": [0.7943478, 0.659659, 0.6961932],
        "std": [0.21563025, 0.24160342, 0.11788896],
        "bw_mean": [0.7041403],
        "bw_std": [0.21754295],
    },
    "kdr": {
        "mean": [0.3200573763148944, 0.2241905006210954, 0.16078055994960383],
        "std": [0.3026714224418703, 0.21879516349046732, 0.17439376177430252],
        "bw_mean": [0.24564162391661948],
        "bw_std": [0.23442767162386982],
    },
}

# Get transformations that are applied to "entire" image
def get_transforms(args, eval, aug):
    trans = []
    
    # Crop full image
    if not eval:
        trans.append(transforms.RandomCrop(args.crop_size, args.padding))
    else:
        trans.append(transforms.CenterCrop(args.crop_size))

    # Resize image after cropping
    if args.image_resize:
        trans.append(transforms.Resize(args.image_resize))

    # Flip image
    if not eval:
        trans.append(transforms.RandomHorizontalFlip())

    # Grayscale, Convert to Tensor and Normalize
    if args.gray:
        trans.append(transforms.Grayscale())
        trans.append(transforms.ToTensor())
        if eval or not args.patch_aug:
            trans.append(transforms.Normalize(mean=aug["bw_mean"], std=aug["bw_std"]))
    else:
        #trans.append(transforms.RandomGrayscale(p=0.25)) # As in CPCV2
        trans.append(transforms.ToTensor())
        if eval or not args.patch_aug:
            trans.append(transforms.Normalize(mean=aug["mean"], std=aug["std"]))

    # If training CPC then patchify, if required also augment
    if not args.fully_supervised:
        if not eval and args.patch_aug:
            trans.append(PatchifyAugment(gray=args.gray, grid_size=args.grid_size))
        else:
            trans.append(Patchify(grid_size=args.grid_size))

    # If patch_aug then normalization goes last
    if not eval and args.patch_aug:
        if args.gray:
            trans.append(PatchAugNormalize(mean=aug["bw_mean"], std=aug["bw_std"]))
        else:
            trans.append(PatchAugNormalize(mean=aug["mean"], std=aug["std"]))

    trans = transforms.Compose(trans)

    s = "Testing" if eval else "Training"
    print(s + ": " + str(trans))

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
        unsupervised_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    # Take subset of training data for train_classifier
    try:
        train_size = args.train_size
        indices = list(range(len(train_dataset)))
        np.random.shuffle(indices)
        train_indices = indices[:train_size]
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers,
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

    elif cifar_classes == 100:
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

        # Noisy Data
        # noise_rate = 0.5
        # noise_type = 'pairflip' # pairflip or symmetric
        # unsupervised_dataset = Noisy_CIFAR100(
        #     root=data_path, train=True, transform=transform_train, download=args.download_dataset, 
        #     noise_type=noise_type, noise_rate=noise_rate
        # )
        # test_dataset = Noisy_CIFAR100(
        #     root=data_path, train=False, transform=transform_valid, download=args.download_dataset, 
        #     noise_type=noise_type, noise_rate=noise_rate
        # )

    else:
        raise Exception("Not a valid number of classes for CIFAR")

    # Get DataLoaders
    unsupervised_loader = torch.utils.data.DataLoader(
        unsupervised_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    # Take subset of training data for train_classifier
    try:
        train_size = args.train_size
        indices = list(range(len(unsupervised_dataset)))
        np.random.shuffle(indices)
        train_indices = indices[:train_size]
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
        train_loader = torch.utils.data.DataLoader(
            unsupervised_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers,
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


def get_medmnist_dataloader(args):
    data_path = os.path.join("data", args.dataset)
    DataClass = getattr(medmnist, medmnist.INFO[args.dataset]['python_class'])

    # Define Transforms
    transform_train = transforms.Compose(
        [get_transforms(args, eval=False, aug=aug[args.dataset])])
    transform_valid = transforms.Compose(
        [get_transforms(args, eval=True, aug=aug[args.dataset])])

    # Get Datasets
    unsupervised_dataset = DataClass(
        root=data_path, split="train", transform=transform_train, download=args.download_dataset, as_rgb=True
    )
    test_dataset = DataClass(
        root=data_path, split="test", transform=transform_valid, download=args.download_dataset, as_rgb=True
    )

    # # Subsetting
    # subset_indices = lambda ds: [idx for idx, (img, label) in enumerate(ds) if label.item() > 0]
    # unsupervised_dataset = torch.utils.data.Subset(unsupervised_dataset, subset_indices(unsupervised_dataset))
    # test_dataset = torch.utils.data.Subset(test_dataset, subset_indices(test_dataset))

    # Get DataLoaders
    unsupervised_loader = torch.utils.data.DataLoader(
        unsupervised_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    # Take subset of training data for train_classifier
    try:
        train_size = args.train_size
        indices = list(range(len(unsupervised_dataset)))
        np.random.shuffle(indices)
        train_indices = indices[:train_size]
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
        train_loader = torch.utils.data.DataLoader(
            # unsupervised_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers,
            unsupervised_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        )
    except AttributeError:  
        # args.train_size is not defined during train_CPC
        # train_loader is not needed during train_CPC
        train_loader = torch.utils.data.DataLoader(
            unsupervised_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        )

    return (unsupervised_loader, train_loader, test_loader)


def get_blood_clusters_dataloader(args):
    data_path = os.path.join("data", "bloodmnist")
    cluster_data_path = os.path.join(data_path, "bloodclusters.csv")

    # Define Transforms
    transform_train = transforms.Compose(
        [get_transforms(args, eval=False, aug=aug[args.dataset])])
    transform_valid = transforms.Compose(
        [get_transforms(args, eval=True, aug=aug[args.dataset])])

    # Get Datasets
    unsupervised_dataset = BloodMNISTClusters(
        root=data_path, split="train", transform=transform_train, download=args.download_dataset, as_rgb=True, cluster_data_path="temp/bloodmnist-unshuffled-tsne-clusters-2.csv", cluster_column='hd-cluster'
    )
    test_dataset = BloodMNISTClusters(
        root=data_path, split="test", transform=transform_valid, download=args.download_dataset, as_rgb=True, cluster_data_path=cluster_data_path, cluster_column='hd-cluster'
    )

    # Get DataLoaders
    unsupervised_loader = torch.utils.data.DataLoader(
        unsupervised_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    # Take subset of training data for train_classifier
    try:
        train_size = args.train_size
        indices = list(range(len(unsupervised_dataset)))
        np.random.shuffle(indices)
        train_indices = indices[:train_size]
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
        train_loader = torch.utils.data.DataLoader(
            # unsupervised_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers,
            unsupervised_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        )
    except AttributeError:  
        # args.train_size is not defined during train_CPC
        # train_loader is not needed during train_CPC
        train_loader = None

    return (unsupervised_loader, train_loader, test_loader)


def get_kdr_dataloader(args):
    data_path = os.path.join("data", args.dataset)

    # Define Transforms
    transform_train = transforms.Compose(
        [get_transforms(args, eval=False, aug=aug[args.dataset])])
    transform_valid = transforms.Compose(
        [get_transforms(args, eval=True, aug=aug[args.dataset])])

    # Get Datasets
    unsupervised_dataset = torchvision.datasets.ImageFolder(
        root=data_path, transform=transform_train
    )
    test_dataset = torchvision.datasets.ImageFolder(
        root=data_path, transform=transform_train
    )

    # Get DataLoaders
    unsupervised_loader = torch.utils.data.DataLoader(
        unsupervised_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    # Take subset of training data for train_classifier
    try:
        train_size = args.train_size
        indices = list(range(len(unsupervised_dataset)))
        np.random.shuffle(indices)
        train_indices = indices[:train_size]
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
        train_loader = torch.utils.data.DataLoader(
            # unsupervised_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers,
            unsupervised_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        )
    except AttributeError:  
        # args.train_size is not defined during train_CPC
        # train_loader is not needed during train_CPC
        train_loader = None

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


def calculate_normalization(dataset):
    if dataset == "stl10":
        data_path = os.path.join("data", "stl10")

        # RGB
        train_transform = transforms.Compose([transforms.ToTensor()])
        train_set = torchvision.datasets.STL10(root=data_path, split="train+unlabeled", download=True, transform=train_transform)

        c1 = np.concatenate([np.asarray(train_set[i][0][0]) for i in range(len(train_set))])  # concatenate each channel
        c2 = np.concatenate([np.asarray(train_set[i][0][1]) for i in range(len(train_set))])
        c3 = np.concatenate([np.asarray(train_set[i][0][2]) for i in range(len(train_set))])

        train_mean = [np.mean(c1, axis=(0, 1,)), np.mean(c2, axis=(0, 1,)), np.mean(c3, axis=(0, 1,))]
        train_std = [np.std(c1, axis=(0, 1)), np.std(c2, axis=(0, 1)), np.std(c3, axis=(0, 1))]

        # grayscale
        train_transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
        train_set = torchvision.datasets.STL10(root=data_path, split="train+unlabeled", download=True, transform=train_transform)

        c = np.concatenate([np.asarray(train_set[i][0][0]) for i in range(len(train_set))])

        gray_train_mean = [np.mean(c, axis=(0, 1,))]
        gray_train_std = [np.std(c, axis=(0, 1))]

        # [0.44087532, 0.42790526, 0.3867924] [0.26826888, 0.2610458, 0.2686684]
        # [0.42709708] [0.257203]

    elif dataset == "cifar10":
        data_path = os.path.join("data", "cifar10")

        # RGB
        train_transform = transforms.Compose([transforms.ToTensor()])
        train_set = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=train_transform)

        c1 = np.concatenate([np.asarray(train_set[i][0][0]) for i in range(len(train_set))])
        c2 = np.concatenate([np.asarray(train_set[i][0][1]) for i in range(len(train_set))])
        c3 = np.concatenate([np.asarray(train_set[i][0][2]) for i in range(len(train_set))])

        train_mean = [np.mean(c1, axis=(0, 1,)), np.mean(c2, axis=(0, 1,)), np.mean(c3, axis=(0, 1,))]
        train_std = [np.std(c1, axis=(0, 1)), np.std(c2, axis=(0, 1)), np.std(c3, axis=(0, 1))]

        # grayscale
        train_transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
        train_set = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=train_transform)

        c = np.concatenate([np.asarray(train_set[i][0][0]) for i in range(len(train_set))])

        gray_train_mean = [np.mean(c, axis=(0, 1,))]
        gray_train_std = [np.std(c, axis=(0, 1))]

        # [0.49139968, 0.48215827, 0.44653124] [0.24703233, 0.24348505, 0.26158768]
        # [0.4808616] [0.23919088]

    elif dataset == "cifar100":
        data_path = os.path.join("data", "cifar100")

        # RGB
        train_transform = transforms.Compose([transforms.ToTensor()])
        train_set = torchvision.datasets.CIFAR100(root=data_path, train=True, download=True, transform=train_transform)

        c1 = np.concatenate([np.asarray(train_set[i][0][0]) for i in range(len(train_set))])
        c2 = np.concatenate([np.asarray(train_set[i][0][1]) for i in range(len(train_set))])
        c3 = np.concatenate([np.asarray(train_set[i][0][2]) for i in range(len(train_set))])

        train_mean = [np.mean(c1, axis=(0, 1,)), np.mean(c2, axis=(0, 1,)), np.mean(c3, axis=(0, 1,))]
        train_std = [np.std(c1, axis=(0, 1)), np.std(c2, axis=(0, 1)), np.std(c3, axis=(0, 1))]

        # grayscale
        train_transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
        train_set = torchvision.datasets.CIFAR100(root=data_path, train=True, download=True, transform=train_transform)

        c = np.concatenate([np.asarray(train_set[i][0][0]) for i in range(len(train_set))])

        gray_train_mean = [np.mean(c, axis=(0, 1,))]
        gray_train_std = [np.std(c, axis=(0, 1))]

        # [0.5070746, 0.48654896, 0.44091788] [0.26733422, 0.25643846, 0.27615058]
        # [0.48748648] [0.25063065]

    elif "mnist" in dataset:
        data_path = os.path.join("data", dataset)
        DataClass = getattr(medmnist, medmnist.INFO[dataset]['python_class'])

        # RGB
        train_transform = transforms.Compose([transforms.ToTensor()])
        train_set = DataClass(root=data_path, split="train", download=True, transform=train_transform, as_rgb=True) # TODO: decide `split`

        c1 = np.concatenate([np.asarray(train_set[i][0][0]) for i in range(len(train_set))])  # concatenate each channel
        c2 = np.concatenate([np.asarray(train_set[i][0][1]) for i in range(len(train_set))])
        c3 = np.concatenate([np.asarray(train_set[i][0][2]) for i in range(len(train_set))])

        train_mean = [np.mean(c1, axis=(0, 1,)), np.mean(c2, axis=(0, 1,)), np.mean(c3, axis=(0, 1,))]
        train_std = [np.std(c1, axis=(0, 1)), np.std(c2, axis=(0, 1)), np.std(c3, axis=(0, 1))]

        # grayscale
        train_transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
        train_set = DataClass(root=data_path, split="train", download=True, transform=train_transform, as_rgb=True) # TODO: decide `split`

        c = np.concatenate([np.asarray(train_set[i][0][0]) for i in range(len(train_set))])

        gray_train_mean = [np.mean(c, axis=(0, 1,))]
        gray_train_std = [np.std(c, axis=(0, 1))]

        # breastmnist
        # [0.3275805, 0.3275805, 0.3275805] [0.20573014, 0.20573014, 0.20573014]
        # [0.3275805] [0.20573014]

        # retinamnist
        # [0.39838937, 0.24472676, 0.15575323] [0.29828647, 0.20048243, 0.15065953]
        # [0.28052062] [0.22027038]

        # bloodmnist
        # [0.7943478, 0.659659, 0.6961932] [0.21563025, 0.24160342, 0.11788896]
        # [0.7041403] [0.21754295]

    elif dataset == "kdr":
        data_path = os.path.join("data", "kdr")

        # RGB
        train_transform = transforms.Compose([transforms.ToTensor()])
        train_set = torchvision.datasets.ImageFolder(root=data_path, transform=train_transform)

        N = len(train_set)

        train_mean = [0, 0, 0]
        train_square_mean = [0, 0, 0]

        for img, lbl in tqdm(train_set):
            for ci in range(3):
                train_mean[ci] += torch.mean(img[ci]).item() / N
                train_square_mean[ci] += torch.mean(img[ci]**2).item() / N

        train_std = [np.sqrt(train_square_mean[ci] - train_mean[ci]**2) for ci in range(3)]

        # c1 = np.concatenate([np.asarray(train_set[i][0][0]) for i in range(len(train_set))])
        # c2 = np.concatenate([np.asarray(train_set[i][0][1]) for i in range(len(train_set))])
        # c3 = np.concatenate([np.asarray(train_set[i][0][2]) for i in range(len(train_set))])

        # train_mean = [np.mean(c1, axis=(0, 1,)), np.mean(c2, axis=(0, 1,)), np.mean(c3, axis=(0, 1,))]
        # train_std = [np.std(c1, axis=(0, 1)), np.std(c2, axis=(0, 1)), np.std(c3, axis=(0, 1))]

        # grayscale
        train_transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
        train_set = torchvision.datasets.ImageFolder(root=data_path, transform=train_transform)

        N = len(train_set)

        gray_train_mean = [0]
        gray_train_square_mean = [0]

        for img, lbl in tqdm(train_set):
            gray_train_mean[0] += torch.mean(img[0]).item() / N
            gray_train_square_mean[0] += torch.mean(img[0]**2).item() / N

        gray_train_std = [np.sqrt(gray_train_square_mean[0] - gray_train_mean[0]**2)]

        # c = np.concatenate([np.asarray(train_set[i][0][0]) for i in range(len(train_set))])

        # gray_train_mean = [np.mean(c, axis=(0, 1,))]
        # gray_train_std = [np.std(c, axis=(0, 1))]

        # [0.3200573763148944, 0.2241905006210954, 0.16078055994960383] [0.3026714224418703, 0.21879516349046732, 0.17439376177430252]
        # [0.24564162391661948] [0.23442767162386982]

    else:
        raise Exception("Not a valid dataset choice")

    print(train_mean, train_std)
    print(gray_train_mean, gray_train_std)


if __name__ == "__main__":
    # calculate_normalization("stl10")
    # calculate_normalization("cifar10")
    # calculate_normalization("cifar100")
    # calculate_normalization("breastmnist")
    # calculate_normalization("retinamnist")
    # calculate_normalization("bloodmnist")
    calculate_normalization("kdr")
