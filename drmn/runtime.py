"""DRMN feature-to-mask runtime. Encoders are frozen and run separately."""
from pathlib import Path
from types import SimpleNamespace
import json
import random
import numpy as np
import torch
import yaml
from drmn.models.head_model.head_net import HeadNet
from drmn.models.head_model.cross_entropy_loss import CrossEntropyLoss
from drmn.models.head_model.dice_loss import DiceLoss


def read_config(path):
    cfg = yaml.safe_load(Path(path).read_text())
    for key in ("num_stages", "num_points", "max_seg_num", "max_sequence_length"):
        if not isinstance(cfg["model"].get(key), int) or cfg["model"][key] <= 0:
            raise ValueError(f"model.{key} must be a positive integer")
    return cfg


def build_model(cfg, device="cpu"):
    c = cfg["model"]
    if c["max_seg_num"] % 2:
        raise ValueError("Original decoder positional encoding requires an even max_seg_num")
    return HeadNet(SimpleNamespace(**c), num_stages=c["num_stages"], num_points=c["num_points"]).to(device)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_checkpoint(model, path):
    from tools.check_checkpoint import inspect_checkpoint
    report = inspect_checkpoint(path)
    if report["status"] != "crc_passed_model_unverified":
        raise ValueError("Checkpoint integrity failed or format unsupported; original weights are required. " + json.dumps(report["errors"][:3]))
    # Tensor-only loading; no unsafe fallback for arbitrary legacy pickle globals.
    import collections
    with torch.serialization.safe_globals([collections.Counter, np.core.multiarray.scalar, np.dtype, type(np.dtype("float64")), type(np.dtype("float32")), type(np.dtype("int64"))]):
        obj = torch.load(path, map_location="cpu", weights_only=True)
    state = obj.get("model_state", obj.get("state_dict", obj))
    state = {k.removeprefix("module."): v for k, v in state.items()}
    if any(not isinstance(v, torch.Tensor) or not torch.isfinite(v).all() for v in state.values()):
        raise ValueError("Checkpoint contains non-tensor or non-finite model parameters")
    model.load_state_dict(state, strict=True)
    return obj


def forward_sample(model, sample):
    selected = int(torch.count_nonzero(sample["noun_ids"]))
    if selected > model.cfg.max_seg_num:
        raise ValueError("Selected noun tokens exceed max_seg_num")
    smallest_sampling_level = sample["fpn"][2].shape[-2] * sample["fpn"][2].shape[-1]
    if model.roi_head.num_stages and model.roi_head.mask_head[0].num_points > smallest_sampling_level:
        raise ValueError("num_points exceeds pixels at sampling level p4; use adequate image size")
    return model(sample["fpn"], sample["language"], sample["noun_ids"])[0]


def stage_loss(predictions, sample):
    if "targets_token" not in sample:
        raise ValueError("Training validation needs targets_token aligned to selected noun tokens")
    n = int(torch.count_nonzero(sample["noun_ids"]))
    if n == 0:
        raise ValueError("No valid noun tokens")
    target = sample["targets_token"][:, :n]
    loss = predictions[0].new_zeros(())
    ce, dice = CrossEntropyLoss(use_sigmoid=True), DiceLoss()
    for prediction in predictions:
        pred = prediction[:, :n].flatten(0, 1)
        gt = target.flatten(0, 1)
        if pred.shape != gt.shape:
            raise ValueError("Target geometry must match output; prepare masks with original preprocessing")
        loss = loss + ce(pred, gt) + dice(pred, gt)
    if not torch.isfinite(loss):
        raise ValueError("Non-finite loss")
    return loss
