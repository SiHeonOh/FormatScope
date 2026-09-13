"""PTQ accuracy for all six configurations (build-plan.md S4.5)."""

import copy
import csv
import datetime
import os
import sys

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "models"))

from data import get_loaders, get_calibration_subset  # noqa: E402
from resnet8 import ResNet8  # noqa: E402

from quant import formats  # noqa: E402
from quant.fakequant import convert_model  # noqa: E402
from quant.calibrate import calibrate, CALIB_IMAGES, CALIB_STAT  # noqa: E402

_CKPT_PATH = os.path.join(_REPO_ROOT, "models", "checkpoints", "fp32.pt")
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
    return 100.0 * correct1 / total, 100.0 * correct5 / total


def _append_csv(fmt, stage, top1, top5):
    os.makedirs(os.path.dirname(_CSV_PATH), exist_ok=True)
    is_new = not os.path.exists(_CSV_PATH) or os.path.getsize(_CSV_PATH) == 0
    with open(_CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(_CSV_COLUMNS)
        writer.writerow([
            fmt, stage, f"{top1:.4f}", f"{top5:.4f}",
            CALIB_IMAGES, CALIB_STAT, True,
            "", 0, "models/checkpoints/fp32.pt",
            datetime.date.today().isoformat(),
        ])


def run_ptq(formats_to_run=formats.FORMAT_IDS, device=None):
    device = device or _select_device()
    print(f"Device: {device}")

    if not os.path.exists(_CKPT_PATH):
        raise FileNotFoundError(
            f"{_CKPT_PATH} not found -- train the FP32 baseline first (models/train.py)"
        )

    base_model = ResNet8(num_classes=10)
    base_model.load_state_dict(torch.load(_CKPT_PATH, map_location="cpu", weights_only=True))

    _, test_loader = get_loaders(batch_size=256)
    calib_batch = get_calibration_subset(n=CALIB_IMAGES, seed=0)

    results = {}
    for fmt in formats_to_run:
        model = copy.deepcopy(base_model)
        convert_model(model, fmt=fmt, first_last=True)
        model.to(device)
        calibrate(model, calib_batch, device=device)

        top1, top5 = _evaluate(model, test_loader, device)
        print(f"{fmt:>10s}  ptq  top1={top1:6.2f}%  top5={top5:6.2f}%")
        _append_csv(fmt, "ptq", top1, top5)
        results[fmt] = top1

    _sanity_check(results)
    return results


def _sanity_check(results):
    """S4.5 watch-out: a scale/rounding bug reads as accuracy near the 10%
    chance line, or as naive INT4 failing to score visibly below INT8."""
    for fmt, top1 in results.items():
        if top1 < 20.0:
            print(f"WARNING: {fmt} PTQ top-1 is {top1:.2f}% -- near chance (10%). "
                  f"Likely a scale or rounding bug, not a real format cost.")
    if "int4" in results and "int8" in results:
        if results["int4"] >= results["int8"] - 1.0:
            print(f"WARNING: int4 ({results['int4']:.2f}%) is not visibly below "
                  f"int8 ({results['int8']:.2f}%) -- the quantizer may be broken.")


if __name__ == "__main__":
    run_ptq()
