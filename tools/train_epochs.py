"""Train DRMN on cached frozen features; supports torchrun and epoch-boundary resume."""
import _bootstrap
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from drmn.runtime import read_config, build_model, seed_all, load_checkpoint, stage_loss
from drmn.datasets.features import load_sample
from drmn.training import dataset_signature, epoch_batches, save_atomic, evaluate_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True)
    parser.add_argument("--data",required=True)
    parser.add_argument("--val-data")
    parser.add_argument("--output",required=True)
    parser.add_argument("--resume")
    parser.add_argument("--epochs",type=int)
    parser.add_argument("--accumulation",type=int,default=1)
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--validate-only",action="store_true",help="One accumulation group forward/backward, no optimizer update or model save")
    a=parser.parse_args()
    cfg=read_config(a.config)
    epochs=a.epochs if a.epochs is not None else cfg["training"].get("epochs",20)
    if epochs<=0 or a.accumulation<=0:parser.error("epochs and accumulation must be positive")
    rank=int(os.environ.get("RANK",0));world=int(os.environ.get("WORLD_SIZE",1))
    if a.device.startswith("cuda"):
        index=int(os.environ.get("LOCAL_RANK",0)) if world>1 else torch.device(a.device).index or 0
        torch.cuda.set_device(index);a.device=f"cuda:{index}"
    if world>1:dist.init_process_group("nccl" if a.device.startswith("cuda") else "gloo")
    try:
        torch.set_num_threads(cfg.get("cpu_threads",2));seed_all(cfg["seed"])
        paths=sorted(Path(a.data).glob("*.npz"))
        val=sorted(Path(a.val_data).glob("*.npz")) if a.val_data else []
        if not paths or (a.val_data and not val):raise ValueError("Empty data split")
        for path in paths:load_sample(path,cfg)
        signature=dataset_signature(paths)
        val_signature=dataset_signature(val) if val else None
        model=build_model(cfg,a.device)
        optimizer=torch.optim.Adam(model.parameters(),lr=cfg["training"]["learning_rate"],weight_decay=cfg["training"].get("weight_decay",0))
        contract={"config":cfg,"world_size":world,"accumulation":a.accumulation,"data_sha256":signature,"val_sha256":val_signature,"device_type":torch.device(a.device).type}
        start=0;step=0;best=-1.;saved_rng=None
        if a.resume:
            checkpoint=load_checkpoint(model,a.resume)
            if checkpoint.get("training_contract")!=contract:raise ValueError("Resume configuration, data, device type or world size changed")
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            start=checkpoint["completed_epochs"];step=checkpoint["global_step"];best=checkpoint["best_score"]
            saved_rng=checkpoint["rng_by_rank"][rank]
        if start>=epochs and not a.validate_only:raise ValueError("Requested epochs already completed")
        if world>1:
            wrapped=DistributedDataParallel(model,device_ids=[torch.device(a.device).index] if a.device.startswith("cuda") else None,find_unused_parameters=True)
        else:wrapped=model
        seed_all(cfg["seed"]+rank)
        if saved_rng:
            torch.set_rng_state(saved_rng["cpu"])
            if a.device.startswith("cuda"):torch.cuda.set_rng_state(saved_rng["cuda"],a.device)
        out=Path(a.output)
        if rank==0:
            out.mkdir(parents=True,exist_ok=True)
            if (out/"last.pth").exists() and not a.resume:raise FileExistsError("Output already contains a run; use --resume or another directory")
            (out/"config.json").write_text(json.dumps(contract,indent=2))
        if world>1:dist.barrier()
        for epoch in range(start,epochs):
            wrapped.train();epoch_loss=0.;seen=0
            groups=epoch_batches(len(paths),cfg["seed"],epoch,rank,world,a.accumulation)
            for group in groups:
                optimizer.zero_grad(set_to_none=True)
                for offset,index in enumerate(group):
                    sample=load_sample(paths[index],cfg,a.device)
                    sync=wrapped.no_sync() if world>1 and offset+1<len(group) else nullcontext()
                    with sync:
                        predictions=wrapped(sample["fpn"],sample["language"],sample["noun_ids"])[0]
                        loss=stage_loss(predictions,sample)
                        (loss/len(group)).backward()
                    epoch_loss+=float(loss.detach());seen+=1
                grads=[p.grad for p in model.parameters() if p.grad is not None]
                if not grads or not all(torch.isfinite(g).all() for g in grads):raise ValueError("Invalid gradients")
                if a.validate_only:
                    if rank==0:(out/"validation.json").write_text(json.dumps({"optimizer_updates":0,"backward_passes_per_rank":len(group),"world_size":world,"loss":epoch_loss/seen},indent=2))
                    return
                optimizer.step();step+=1
            totals=torch.tensor([epoch_loss,seen],dtype=torch.float64,device=a.device)
            if world>1:dist.all_reduce(totals)
            rng={"cpu":torch.get_rng_state(),"cuda":torch.cuda.get_rng_state(a.device) if a.device.startswith("cuda") else None}
            rngs=[None]*world
            if world>1:dist.all_gather_object(rngs,rng)
            else:rngs=[rng]
            if rank==0:
                metrics=evaluate_files(model,val,cfg,a.device) if val else None
                score=metrics["overall"]["average_recall"] if metrics else None
                improved=score is not None and score>best
                if improved:best=score
                payload={"model_state":model.state_dict(),"optimizer_state":optimizer.state_dict(),"completed_epochs":epoch+1,"global_step":step,"best_score":best,"training_contract":contract,"config":cfg,"rng_by_rank":rngs}
                save_atomic(payload,out/"last.pth")
                if improved:save_atomic(payload,out/"best.pth")
                row={"epoch":epoch+1,"global_step":step,"loss":float(totals[0]/totals[1]),"learning_rate":optimizer.param_groups[0]["lr"],"validation":metrics,"samples_including_distributed_padding":int(totals[1])}
                with (out/"history.jsonl").open("a") as stream:stream.write(json.dumps(row)+"\n")
                print(json.dumps(row),flush=True)
            if world>1:dist.barrier()
    finally:
        if dist.is_initialized():dist.destroy_process_group()

if __name__=="__main__":main()
