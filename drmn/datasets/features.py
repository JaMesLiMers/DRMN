"""Validated, pickle-free frozen-feature interface; one image per NPZ."""
from pathlib import Path
import numpy as np
import torch


def load_sample(path, cfg, device="cpu"):
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    keys = ["fpn_p2", "fpn_p3", "fpn_p4", "fpn_p5", "language", "noun_ids"]
    for key in keys:
        if key not in data or not np.isfinite(data[key]).all():
            raise ValueError(f"Missing or non-finite feature: {key}")
    tensor = lambda a, dtype=torch.float32: torch.as_tensor(a, dtype=dtype, device=device)
    fpn = [tensor(data[k]).unsqueeze(0) for k in keys[:4]]
    if any(x.ndim != 4 or x.shape[1] != 256 for x in fpn):
        raise ValueError("FPN features must have shape [256,H,W]")
    h, w = fpn[0].shape[-2:]
    if h % 8 or w % 8 or any(tuple(x.shape[-2:]) != (h//2**i,w//2**i) for i,x in enumerate(fpn)):
        raise ValueError("Expected p2/p3/p4/p5 pyramid from image padded to multiples of 32")
    language = tensor(data["language"]).unsqueeze(0)
    noun_ids = tensor(data["noun_ids"], torch.long).unsqueeze(0)
    if language.shape != (1,cfg["model"]["max_sequence_length"],768) or noun_ids.shape != language.shape[:2]:
        raise ValueError("Language or noun sequence shape does not match config")
    if np.any(data["noun_ids"] != data["noun_ids"].astype(np.int64)) or (noun_ids < 0).any():
        raise ValueError("noun_ids must be nonnegative integers")
    selected = int(torch.count_nonzero(noun_ids))
    if not 0 < selected <= cfg["model"]["max_seg_num"]:
        raise ValueError("Invalid selected noun count")
    result = {"id": Path(path).stem, "fpn":fpn,"language":language,"noun_ids":noun_ids}
    for key in ("targets_token", "targets_phrase"):
        if key in data:
            if not np.isfinite(data[key]).all() or not np.isin(data[key],[0,1]).all():
                raise ValueError(f"{key} must contain finite binary masks")
            result[key] = tensor(data[key]).unsqueeze(0)
    if "targets_token" in result and result["targets_token"].shape[1] < selected:
        raise ValueError("Not enough token targets")
    for key in ("phrase_intervals", "ann_types", "ann_categories"):
        if key in data:
            if not np.isfinite(data[key]).all() or np.any(data[key] != data[key].astype(np.int64)):
                raise ValueError(f"{key} must contain integers")
            result[key] = data[key].astype(np.int64)
    if "phrase_intervals" in result:
        intervals=result["phrase_intervals"]
        if intervals.ndim!=1 or len(intervals)<2 or intervals[0]!=0 or intervals[-1]!=selected or np.any(np.diff(intervals)<=0):
            raise ValueError("phrase_intervals must partition all selected noun tokens")
        for key in ("ann_types","ann_categories"):
            if key in result and (result[key].shape!=(len(intervals)-1,) or not np.isin(result[key],[1,2]).all()):
                raise ValueError(f"Invalid phrase groups: {key}")
    return result
