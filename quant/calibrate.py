"""Activation-range calibration from a fixed image subset (build-plan.md S4.4)."""

import numpy as np
import torch

from quant.fakequant import QConv2d, QLinear, _BLOCK_FORMATS, _ELEMENT_MAX

CALIB_IMAGES = 512
CALIB_STAT = "p99.99"


def calibrate(model, calib_batch, device="cpu"):
    """Run `calib_batch` through `model` once and set a fixed per-tensor
    activation scale (99.99th percentile of |x|, decision D10) on every
    elementwise-format QConv2d/QLinear. Block-format (MX) modules are left
    alone: their block scales are always computed fresh on the fly (S4.4).
    """
    targets = [m for m in model.modules()
               if isinstance(m, (QConv2d, QLinear)) and m.fmt not in _BLOCK_FORMATS and m.fmt is not None]

    observed = {}

    def make_hook(module):
        def hook(mod, inputs):
            observed[module] = inputs[0].detach()
        return hook

    handles = [m.register_forward_pre_hook(make_hook(m)) for m in targets]

    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(calib_batch.to(device))
    model.train(was_training)

    for h in handles:
        h.remove()

    for m in targets:
        x = observed[m].cpu().double().numpy()
        amax = np.percentile(np.abs(x), 99.99)
        amax = amax if amax > 0 else 1.0
        m._act_scale = np.array(amax / _ELEMENT_MAX[m.fmt])

    return model
