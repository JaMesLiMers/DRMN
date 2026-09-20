"""Bounded feature-to-mask training or backward validation; never starts an epoch loop."""
import _bootstrap
import argparse,json
from pathlib import Path
import torch
from drmn.runtime import read_config,build_model,seed_all,load_checkpoint,forward_sample,stage_loss
from drmn.datasets.features import load_sample


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",required=True);p.add_argument("--data",required=True)
    p.add_argument("--checkpoint");p.add_argument("--device",default="cpu")
    p.add_argument("--max-steps",type=int,default=0,help="0: backward validation only, no optimizer update")
    p.add_argument("--output",default="artifacts/runs/validation")
    a=p.parse_args()
    if a.max_steps<0:p.error("max-steps must be nonnegative")
    cfg=read_config(a.config);seed_all(cfg["seed"]);torch.set_num_threads(cfg.get("cpu_threads",2))
    model=build_model(cfg,a.device)
    checkpoint=load_checkpoint(model,a.checkpoint) if a.checkpoint else None
    model.train()
    files=sorted(Path(a.data).glob("*.npz"))
    if not files:raise ValueError("No NPZ samples")
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg["training"]["learning_rate"])
    if checkpoint and "optimizer_state" in checkpoint:optimizer.load_state_dict(checkpoint["optimizer_state"])
    history=[]
    for step in range(max(1,a.max_steps)):
        sample=load_sample(files[step%len(files)],cfg,a.device)
        optimizer.zero_grad(set_to_none=True)
        loss=stage_loss(forward_sample(model,sample),sample)
        loss.backward()
        grads=[v.grad for v in model.parameters() if v.grad is not None]
        if not grads or not all(torch.isfinite(g).all() for g in grads):raise ValueError("Invalid gradients")
        if a.max_steps:optimizer.step()
        history.append({"step":step,"loss":float(loss.detach()),"updated":bool(a.max_steps)})
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    report={"history":history,"torch":torch.__version__,"config":cfg,"validation_only":not bool(a.max_steps),"weight_source":a.checkpoint or "random initialization, not paper weights"}
    (out/"run.json").write_text(json.dumps(report,indent=2))
    if a.max_steps:
        torch.save({"model_state":model.state_dict(),"optimizer_state":optimizer.state_dict(),"config":cfg,"steps":a.max_steps},out/"last.pth")
    print(json.dumps(report,indent=2))

if __name__=="__main__":main()
