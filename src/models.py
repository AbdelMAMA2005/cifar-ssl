import torch
import torch.nn as nn
import torchvision.models as tvm

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2435, 0.2616)

def resnet18_cifar(num_classes=10):
    """ResNet-18 adapted for 32x32 CIFAR (3x3 stem, no maxpool)."""
    m = tvm.resnet18(num_classes=num_classes)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m

class Encoder(nn.Module):
    """ResNet-18 backbone without classifier. Returns 512-d features."""
    def __init__(self):
        super().__init__()
        base = resnet18_cifar()
        self.features = nn.Sequential(*list(base.children())[:-1])  # conv1..avgpool
        self.out_dim = 512

    def forward(self, x):
        return torch.flatten(self.features(x), 1)