import os
import numpy as np
import torch
from torchvision import datasets, transforms

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _default_data_dir():
    return os.environ.get("FORMATSCOPE_DATA", os.path.join(_REPO_ROOT, "data"))


def get_loaders(batch_size=128, data_dir=None, seed=0):
    if data_dir is None:
        data_dir = _default_data_dir()

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)

    g = torch.Generator().manual_seed(seed)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, generator=g
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    return train_loader, test_loader


def get_calibration_subset(n=512, seed=0, data_dir=None):
    if data_dir is None:
        data_dir = _default_data_dir()

    # test-style transform (no augmentation) applied to training data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=n, replace=False)
    images = torch.stack([dataset[int(i)][0] for i in indices])
    return images
