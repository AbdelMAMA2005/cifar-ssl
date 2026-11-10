import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from src.models import CIFAR_MEAN, CIFAR_STD

def train_transform():
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

def test_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

def get_cifar10(root, batch_size=128, num_workers=4):
    tr = datasets.CIFAR10(root, train=True, download=True, transform=train_transform())
    te = datasets.CIFAR10(root, train=False, download=True, transform=test_transform())
    tr_loader = DataLoader(tr, batch_size, shuffle=True, num_workers=num_workers,
                           pin_memory=True, drop_last=True)
    te_loader = DataLoader(te, batch_size, shuffle=False, num_workers=num_workers,
                           pin_memory=True)
    return tr_loader, te_loader, tr, te

class CIFAR10C(Dataset):
    """Expects: {root}/labels.npy and {root}/{corruption}.npy (Zenodo 2535967)."""
    def __init__(self, root, corruption, severity, transform=None):
        data = np.load(f"{root}/{corruption}.npy")
        labels = np.load(f"{root}/labels.npy")
        idx = slice((severity - 1) * 10000, severity * 10000)
        self.data = data[idx]
        self.labels = labels[idx]
        self.transform = transform or test_transform()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        img = Image.fromarray(self.data[i])
        return self.transform(img), int(self.labels[i])