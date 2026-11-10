import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from src.models import CIFAR_MEAN, CIFAR_STD

class SimCLR(nn.Module):
    def __init__(self, encoder, feat_dim=512, proj_dim=128):
        super().__init__()
        self.encoder = encoder
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, feat_dim), nn.ReLU(inplace=True),
            nn.Linear(feat_dim, proj_dim),
        )

    def forward(self, x):
        h = self.encoder(x)
        return F.normalize(self.proj(h), dim=1)

def nt_xent(z, temperature=0.5):
    """z: (2B, D); pairs are (i, i+B)."""
    N = z.size(0)
    sim = z @ z.t() / temperature
    sim = sim.masked_fill(torch.eye(N, dtype=torch.bool, device=z.device), -1e9)
    targets = (torch.arange(N, device=z.device) + N // 2) % N
    return F.cross_entropy(sim, targets)

def simclr_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])