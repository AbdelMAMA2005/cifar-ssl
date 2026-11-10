import argparse, os
import numpy as np
import torch
import torch.nn as nn
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

from src.data import test_transform
from src.models import Encoder, resnet18_cifar

@torch.no_grad()
def extract_features(enc, loader, device):
    feats, labels = [], []
    enc.eval()
    for x, y in loader:
        feats.append(enc(x.to(device)).cpu())
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)

def load_encoder(ckpt_path, kind, device):
    if kind == "supervised":
        m = resnet18_cifar()
        m.load_state_dict(torch.load(ckpt_path, map_location=device))
        base = nn.Sequential(*list(m.children())[:-1])
    else:
        base = Encoder()
        base.load_state_dict(torch.load(ckpt_path, map_location=device))

    class _Enc(nn.Module):
        def __init__(self, b):
            super().__init__()
            self.b = b
        def forward(self, x):
            return torch.flatten(self.b(x), 1)

    return _Enc(base).to(device)

def train_linear_probe(Xtr, ytr, Xte, yte, device, epochs=100, lr=1e-3, wd=0.0):
    clf = nn.Linear(Xtr.size(1), 10).to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=wd)
    crit = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=256, shuffle=True)
    for _ in range(epochs):
        for x, y in loader:
            loss = crit(clf(x.to(device)), y.to(device))
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        acc = (clf(Xte.to(device)).argmax(1) == yte.to(device)).float().mean().item()
    return clf, acc

def knn_probe(Xtr, ytr, Xte, yte, k=20):
    knn = KNeighborsClassifier(n_neighbors=k, metric="cosine", n_jobs=-1)
    knn.fit(Xtr.numpy(), ytr.numpy())
    return (knn.predict(Xte.numpy()) == yte.numpy()).mean()

def main(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tr = datasets.CIFAR10(cfg["data_root"], train=True,  download=True, transform=test_transform())
    te = datasets.CIFAR10(cfg["data_root"], train=False, download=True, transform=test_transform())
    tr_loader = DataLoader(tr, 256, shuffle=False, num_workers=4)
    te_loader = DataLoader(te, 256, shuffle=False, num_workers=4)

    enc = load_encoder(cfg["ckpt"], cfg["encoder_kind"], device)
    Xtr, ytr = extract_features(enc, tr_loader, device)
    Xte, yte = extract_features(enc, te_loader, device)

    clf, lin_acc = train_linear_probe(Xtr, ytr, Xte, yte, device)
    knn_acc = knn_probe(Xtr, ytr, Xte, yte, k=20)

    print(f"Linear probe acc: {lin_acc:.4f}")
    print(f"kNN (k=20) acc : {knn_acc:.4f}")

    os.makedirs(cfg["out_dir"], exist_ok=True)
    torch.save(clf.state_dict(), os.path.join(cfg["out_dir"], "linear_clf.pt"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    main(cfg)