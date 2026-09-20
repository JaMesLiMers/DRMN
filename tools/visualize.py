"""Predict phrase masks from a frozen-feature sample and a strict checkpoint."""
import _bootstrap
import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from drmn.runtime import read_config,build_model,load_checkpoint,forward_sample
from drmn.datasets.features import load_sample
from drmn.evaluation.metrics import phrase_masks

p=argparse.ArgumentParser(description=__doc__)
p.add_argument("--config",required=True);p.add_argument("--sample",required=True);p.add_argument("--checkpoint",required=True)
p.add_argument("--output",default="artifacts/predictions");p.add_argument("--device",default="cpu")
if __name__=="__main__":
 a=p.parse_args();cfg=read_config(a.config);torch.set_num_threads(cfg.get("cpu_threads",2));m=build_model(cfg,a.device);load_checkpoint(m,a.checkpoint);m.eval()
 sample=load_sample(a.sample,cfg,a.device)
 if "phrase_intervals" not in sample:raise ValueError("Missing phrase intervals")
 with torch.inference_mode():masks=phrase_masks(forward_sample(m,sample)[-1],sample["phrase_intervals"])
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 for i,mask in enumerate(masks):Image.fromarray(mask.cpu().numpy().astype(np.uint8)*255).save(out/f"phrase_{i:03d}.png")
 print(f"Saved {len(masks)} masks to {out}")
