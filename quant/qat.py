"""5-epoch quantization-aware fine-tune for all six configurations (build-plan.md S4.6)."""

import copy
import csv
import datetime
import os
import sys

import numpy as np
import torch
import torch.nn as nn

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "models"))

from data import get_loaders, get_calibration_subset  # noqa: E402
from resnet8 import ResNet8  # noqa: E402

from quant.fakequant import convert_model  # noqa: E402
from quant.calibrate import calibrate, CALIB_IMAGES, CALIB_STAT  # noqa: E402

_CKPT_PATH = os.path.join(_REPO_ROOT, "models", "checkpoints", "fp32.pt")
_CSV_PATH = os.path.join(_REPO_ROOT, "results", "accuracy.csv")
_CSV_COLUMNS = [
    "format", "stage", "top1", "top5",
    "calib_images", "calib_stat", "quantize_first_last",
    "epochs", "seed", "checkpoint", "date",
]

# D12/H2: most informative rows land first.
_RUN_ORDER = ("int4", "mxfp4", "int8", "fp8e4m3", "mxint8", "int4_b32")
_EPOCHS = int(os.environ.get("QAT_EPOCHS", 5))


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
    return 100.0 * correct1 / total, 100.0 * correct5 / total


def _append_csv(fmt, top1, top5, epochs):
    os.makedirs(os.path.dirname(_CSV_PATH), exist_ok=True)
    is_new = not os.path.exists(_CSV_PATH) or os.path.getsize(_CSV_PATH) == 0
    with open(_CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(_CSV_COLUMNS)
        writer.writerow([
            fmt, "qat", f"{top1:.4f}", f"{top5:.4f}",
            CALIB_IMAGES, CALIB_STAT, True,
            epochs, 0, "models/checkpoints/fp32.pt",
            datetime.date.today().isoformat(),
        ])


def _read_ptq_rows():
    if not os.path.exists(_CSV_PATH):
        return {}
    out = {}
    with open(_CSV_PATH) as f:
        for row in csv.DictReader(f):
            if row["stage"] == "ptq":
                out[row["format"]] = float(row["top1"])
    return out


def _h2_check(qat_results, ptq_results):
    """H2: does QAT close most of MXFP4's (and MXINT8's) PTQ-stage gap to INT8?"""
    if "int8" not in ptq_results:
        return
    for fmt in ("mxfp4", "int4_b32", "mxint8"):
        if fmt not in qat_results or fmt not in ptq_results:
            continue
        gap_before = ptq_results["int8"] - ptq_results[fmt]
        gap_after = qat_results.get("int8", ptq_results["int8"]) - qat_results[fmt]
        closed = 100.0 * (1 - gap_after / gap_before) if gap_before else float("nan")
        print(f"H2 [{fmt}]: PTQ gap to int8 was {gap_before:.2f}pp, "
              f"QAT gap is {gap_after:.2f}pp ({closed:.0f}% closed)")


def run_qat(formats_to_run=_RUN_ORDER, device=None, epochs=_EPOCHS):
    device = device or _select_device()
    print(f"Device: {device}, epochs: {epochs}")

    if not os.path.exists(_CKPT_PATH):
        raise FileNotFoundError(f"{_CKPT_PATH} not found -- train the FP32 baseline first")

    base_model = ResNet8(num_classes=10)
    base_model.load_state_dict(torch.load(_CKPT_PATH, map_location="cpu", weights_only=True))

    train_loader, test_loader = get_loaders(batch_size=128, seed=0)
    calib_batch = get_calibration_subset(n=CALIB_IMAGES, seed=0)
    ptq_results = _read_ptq_rows()

    results = {}
    for fmt in formats_to_run:
        torch.manual_seed(0)
        np.random.seed(0)

        model = copy.deepcopy(base_model)
        convert_model(model, fmt=fmt, first_last=True)
        model.to(device)
        calibrate(model, calib_batch, device=device)  # fixed act scale; MX recomputes per-block anyway

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(1, epochs + 1):
            model.train()
            train_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                loss = criterion(model(images), labels)
                loss.backward()
                # Folded-BN fine-tuning diverges within a handful of steps without this
                # (verified at fmt=None too, so it's not a quantization bug -- see PR).
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item() * labels.size(0)
            train_loss /= len(train_loader.dataset)
            scheduler.step()
            print(f"{fmt:>10s}  epoch {epoch}/{epochs}  train_loss={train_loss:.4f}")

        top1, top5 = _evaluate(model, test_loader, device)
        print(f"{fmt:>10s}  qat  top1={top1:6.2f}%  top5={top5:6.2f}%")
        _append_csv(fmt, top1, top5, epochs)
        results[fmt] = top1

    _h2_check(results, ptq_results)
    return results


if __name__ == "__main__":
    run_qat()
