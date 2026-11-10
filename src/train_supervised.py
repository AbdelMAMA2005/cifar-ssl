import argparse, os, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from src.data import get_cifar10
from src.models import resnet18_cifar

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    crit = nn.CrossEntropyLoss()
    correct = total = 0
    loss_sum = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss_sum += crit(out, y).item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return correct / total, loss_sum / total

def main(cfg):
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tr_loader, te_loader, _, _ = get_cifar10(cfg["data_root"], cfg["batch_size"], cfg["num_workers"])

    model = resnet18_cifar().to(device)
    crit = nn.CrossEntropyLoss()
    if cfg["optimizer"] == "sgd":
        opt = optim.SGD(model.parameters(), lr=cfg["lr"], momentum=0.9,
                        weight_decay=cfg["wd"], nesterov=True)
    else:
        opt = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = CosineAnnealingLR(opt, T_max=cfg["epochs"])

    os.makedirs(cfg["out_dir"], exist_ok=True)
    writer = SummaryWriter(os.path.join(cfg["out_dir"], "tb"))
    best = 0.0

    for ep in range(cfg["epochs"]):
        model.train()
        t0 = time.time()
        tot_loss = tot_n = tot_c = 0
        for x, y in tr_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = crit(out, y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot_loss += loss.item() * x.size(0)
            tot_c += (out.argmax(1) == y).sum().item()
            tot_n += x.size(0)
        sched.step()
        val_acc, val_loss = evaluate(model, te_loader, device)
        writer.add_scalar("train/loss", tot_loss / tot_n, ep)
        writer.add_scalar("train/acc",  tot_c / tot_n, ep)
        writer.add_scalar("val/loss",   val_loss, ep)
        writer.add_scalar("val/acc",    val_acc, ep)
        writer.add_scalar("lr",         sched.get_last_lr()[0], ep)
        print(f"ep {ep+1:03d}/{cfg['epochs']} | tr_loss {tot_loss/tot_n:.4f} "
              f"tr_acc {tot_c/tot_n:.4f} | val_loss {val_loss:.4f} "
              f"val_acc {val_acc:.4f} | {time.time()-t0:.1f}s")
        if val_acc > best:
            best = val_acc
            torch.save(model.state_dict(), os.path.join(cfg["out_dir"], "best.pt"))
    writer.close()
    print(f"Best val acc: {best:.4f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    main(cfg)