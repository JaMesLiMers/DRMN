"""Legacy PNG mask aggregation and discrete Average Recall protocol."""
import numpy as np
import torch


def average_recall(ious):
    a = np.asarray(ious, dtype=float)
    if a.size == 0:
        return None
    if not np.isfinite(a).all() or np.any((a<0)|(a>1)):
        raise ValueError("IoU must be finite and in [0,1]")
    # Original integration uses thresholds[:-1], inclusive IoU >= t.
    thresholds = np.arange(0,1,0.0001)
    area = np.r_[0,np.cumsum(np.diff(thresholds))]
    return float(area[np.searchsorted(thresholds[:-1],a,side="right")].mean())


def phrase_masks(logits, intervals):
    probabilities = logits.sigmoid()[0]
    return torch.stack([probabilities[int(a):int(b)].mean(0)>0.5 for a,b in zip(intervals[:-1],intervals[1:])])


def sample_ious(logits, sample):
    for key in ("phrase_intervals","targets_phrase","ann_types","ann_categories"):
        if key not in sample: raise ValueError(f"Evaluation requires {key}")
    pred = phrase_masks(logits, sample["phrase_intervals"])
    gt = sample["targets_phrase"][0].bool()
    if pred.shape != gt.shape:
        raise ValueError("Prediction/target geometry mismatch; no silent resize in evaluation")
    union = (pred | gt).flatten(1).sum(1)
    if (union==0).any(): raise ValueError("Undefined empty-union IoU")
    return ((pred & gt).flatten(1).sum(1)/union).detach().cpu().numpy()


def summarize(ious, types, categories):
    ious,types,categories=map(np.asarray,(ious,types,categories))
    groups={"overall":np.ones(len(ious),dtype=bool),"singulars":types==1,"plurals":types==2,"things":categories==1,"stuff":categories==2}
    return {key:{"count":int(mask.sum()),"average_recall":average_recall(ious[mask])} for key,mask in groups.items()}
