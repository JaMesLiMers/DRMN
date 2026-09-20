"""Evaluate all supplied frozen-feature samples using a strict checkpoint."""
import _bootstrap
import argparse,json
from pathlib import Path
import torch
from drmn.runtime import read_config,build_model,load_checkpoint,forward_sample
from drmn.datasets.features import load_sample
from drmn.evaluation.metrics import sample_ious,summarize


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",required=True);p.add_argument("--data",required=True)
    p.add_argument("--checkpoint",required=True);p.add_argument("--device",default="cpu")
    p.add_argument("--output",default="artifacts/runs/evaluation.json")
    a=p.parse_args();cfg=read_config(a.config);torch.set_num_threads(cfg.get("cpu_threads",2))
    m=build_model(cfg,a.device);load_checkpoint(m,a.checkpoint);m.eval()
    files=sorted(Path(a.data).glob("*.npz"))
    if not files:raise ValueError("No samples")
    values=[];types=[];categories=[]
    with torch.inference_mode():
        for path in files:
            sample=load_sample(path,cfg,a.device)
            ious=sample_ious(forward_sample(m,sample)[-1],sample)
            values.extend(ious.tolist());types.extend(sample["ann_types"].tolist());categories.extend(sample["ann_categories"].tolist())
    source_path=Path(a.data)/"SOURCE.json"
    provenance=json.loads(source_path.read_text()) if source_path.exists() else {"verified":False}
    report={"dataset_provenance":provenance,"metrics":summarize(values,types,categories),"images":len(files),"checkpoint":str(Path(a.checkpoint).resolve()),"config":cfg,"protocol":"legacy discrete AR; prepared masks at model output geometry"}
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=="__main__":main()
