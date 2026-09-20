"""Epoch training utilities with deterministic epoch-boundary resume."""
import hashlib
import json
import os
from pathlib import Path
import torch
from drmn.runtime import forward_sample, stage_loss
from drmn.datasets.features import load_sample
from drmn.evaluation.metrics import sample_ious, summarize


def dataset_signature(paths):
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(4*1024*1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def epoch_batches(size, seed, epoch, rank, world_size, accumulation):
    if size <= 0 or accumulation <= 0:
        raise ValueError("Dataset and accumulation must be positive")
    indices = torch.randperm(size, generator=torch.Generator().manual_seed(seed+epoch)).tolist()
    # Match DistributedSampler: pad explicitly to equal rank lengths.
    total = ((size+world_size-1)//world_size)*world_size
    indices = (indices*((total+size-1)//size))[:total][rank:total:world_size]
    return [indices[i:i+accumulation] for i in range(0,len(indices),accumulation)]


def save_atomic(payload, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


@torch.no_grad()
def evaluate_files(model, paths, cfg, device):
    was_training = model.training
    model.eval()
    values, types, categories = [], [], []
    for path in paths:
        sample = load_sample(path, cfg, device)
        values.extend(sample_ious(forward_sample(model, sample)[-1],sample).tolist())
        types.extend(sample["ann_types"].tolist())
        categories.extend(sample["ann_categories"].tolist())
    model.train(was_training)
    return summarize(values, types, categories)
