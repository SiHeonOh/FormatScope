import csv
import datetime
import os

import numpy as np
import torch
import torch.nn as nn

from data import get_loaders
from resnet8 import ResNet8

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CKPT_DIR = os.path.join(_REPO_ROOT, "models", "checkpoints")
_CSV_PATH = os.path.join(_REPO_ROOT, "results", "accuracy.csv")
_CSV_COLUMNS = [
    "format", "stage", "top1", "top5",
    "calib_images", "calib_stat", "quantize_first_last",
    "epochs", "seed", "checkpoint", "date",
]


def _select_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _evaluate(model, loader, device):
    model.eval()
    correct1 = correct5 = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            _, pred5 = logits.topk(5, dim=1)
            correct1 += (pred5[:, 0] == labels).sum().item()
            correct5 += (labels.unsqueeze(1).eq(pred5)).any(dim=1).sum().item()
            total += labels.size(0)
    top1 = 100.0 * correct1 / total
    top5 = 100.0 * correct5 / total
    return top1, top5


def _append_csv(top1, top5, epochs, seed, checkpoint):
    os.makedirs(os.path.dirname(_CSV_PATH), exist_ok=True)
    is_new = not os.path.exists(_CSV_PATH) or os.path.getsize(_CSV_PATH) == 0
    with open(_CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(_CSV_COLUMNS)
        writer.writerow([
            "fp32", "none", f"{top1:.4f}", f"{top5:.4f}",
            "", "", "",
            epochs, seed, checkpoint,
            datetime.date.today().isoformat(),
        ])


def main():
    # SEED=n varies the run for the multi-seed sweep (scripts/seed_sweep.sh).
    # Seed 0 keeps the plain checkpoint name so quant/eval.py sees no change.
    seed = int(os.environ.get("SEED", 0))
    torch.manual_seed(seed)
    np.random.seed(seed)

    epochs = int(os.environ.get("EPOCHS", 60))
    device = _select_device()
    print(f"Device: {device}, epochs: {epochs}, seed: {seed}")

    os.makedirs(_CKPT_DIR, exist_ok=True)
    stem = "fp32" if seed == 0 else f"fp32_seed{seed}"
    ckpt_last = os.path.join(_CKPT_DIR, f"{stem}.pt")
    ckpt_best = os.path.join(_CKPT_DIR, f"{stem}_best.pt")

    train_loader, test_loader = get_loaders(batch_size=128, seed=seed)

    model = ResNet8(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_top1 = 0.0
    final_top1 = final_top5 = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * labels.size(0)
        train_loss /= len(train_loader.dataset)
        scheduler.step()

        top1, top5 = _evaluate(model, test_loader, device)
        print(f"Epoch {epoch:3d}  train_loss={train_loss:.4f}  test_top1={top1:.2f}%")

        torch.save(model.state_dict(), ckpt_last)
        if top1 > best_top1:
            best_top1 = top1
            torch.save(model.state_dict(), ckpt_best)

        final_top1, final_top5 = top1, top5

    _append_csv(final_top1, final_top5, epochs, seed=seed,
                checkpoint=f"models/checkpoints/{stem}.pt")
    print(f"Done. Final top-1: {final_top1:.2f}%  top-5: {final_top5:.2f}%")


if __name__ == "__main__":
    main()
