import argparse, os, random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets

from src.models import Encoder
from src.simclr import SimCLR, nt_xent, simclr_transform

class TwoViewCIFAR(Dataset):
    def __init__(self, root, tf):
        self.ds = datasets.CIFAR10(root, train=True, download=True)
        self.tf = tf
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, i):
        img, _ = self.ds[i]
        return self.tf(img), self.tf(img)

def main(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"]); random.seed(cfg["seed"])

    ds = TwoViewCIFAR(cfg["data_root"], simclr_transform())
    dl = DataLoader(ds, cfg["batch_size"], shuffle=True, num_workers=4,
                    drop_last=True, pin_memory=True)

    model = SimCLR(Encoder()).to(device)
    opt = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])

    os.makedirs(cfg["out_dir"], exist_ok=True)
    writer = SummaryWriter(os.path.join(cfg["out_dir"], "tb"))
    step = 0

    for ep in range(cfg["epochs"]):
        model.train()
        for v1, v2 in dl:
            x = torch.cat([v1, v2], dim=0).to(device)
            loss = nt_xent(model(x), cfg["temperature"])
            opt.zero_grad(); loss.backward(); opt.step()
            writer.add_scalar("simclr/loss", loss.item(), step); step += 1
        sched.step()
        print(f"epoch {ep+1:03d}/{cfg['epochs']} loss {loss.item():.4f}")
        if (ep + 1) % cfg["save_every"] == 0:
            torch.save(model.encoder.state_dict(),
                       os.path.join(cfg["out_dir"], f"encoder_ep{ep+1}.pt"))
    torch.save(model.encoder.state_dict(), os.path.join(cfg["out_dir"], "encoder.pt"))
    writer.close()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    main(cfg)