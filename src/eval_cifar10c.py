import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data import CIFAR10C
from src.models import Encoder, resnet18_cifar

CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness", "contrast",
    "elastic_transform", "pixelate", "jpeg_compression",
]

def build_predict_fn(cfg, device):
    if cfg["encoder_kind"] == "supervised":
        m = resnet18_cifar().to(device)
        m.load_state_dict(torch.load(cfg["ckpt"], map_location=device))
        m.eval()
        return lambda x: m(x)
    else:
        enc = Encoder().to(device)
        enc.load_state_dict(torch.load(cfg["ckpt"], map_location=device))
        clf = nn.Linear(512, 10).to(device)
        clf.load_state_dict(torch.load(cfg["linear_ckpt"], map_location=device))
        enc.eval(); clf.eval()
        return lambda x: clf(enc(x))

def main(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predict = build_predict_fn(cfg, device)

    accs = {}
    with torch.no_grad():
        for c in CORRUPTIONS:
            row = []
            for sev in range(1, 6):
                ds = CIFAR10C(cfg["cifar10c_root"], c, sev)
                dl = DataLoader(ds, 256, shuffle=False, num_workers=4)
                correct = total = 0
                for x, y in dl:
                    x, y = x.to(device), y.to(device)
                    correct += (predict(x).argmax(1) == y).sum().item()
                    total += x.size(0)
                row.append(correct / total)
            accs[c] = row
            print(f"{c:20s} mean={sum(row)/5:.4f} | " + " ".join(f"{a:.3f}" for a in row))

    mca = sum(sum(r) for r in accs.values()) / (len(accs) * 5)
    print(f"\nMean corruption accuracy (mCA): {mca:.4f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    main(cfg)