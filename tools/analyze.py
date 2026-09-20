"""Export refinement, top-k attention and deformable sampling diagnostics."""
if __package__ in (None, ""):
    import _bootstrap
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from drmn.runtime import read_config, build_model, load_checkpoint
from drmn.datasets.features import load_sample


def export_analysis(predictions, diagnostics, sample, output, phrase=0):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    intervals=sample["phrase_intervals"]
    if phrase<0 or phrase>=len(intervals)-1:raise ValueError("Phrase index out of range")
    begin,end=map(int,intervals[phrase:phrase+2])
    arrays={};fig,axes=plt.subplots(1,len(predictions),figsize=(4*len(predictions),4),squeeze=False)
    for index,pred in enumerate(predictions):
        probability=pred.sigmoid()[0,begin:end].mean(0).detach().cpu().numpy()
        arrays[f"stage_{index}_probability"]=probability
        axes[0,index].imshow(probability,vmin=0,vmax=1,cmap="viridis")
        axes[0,index].set_title(f"Initial" if index==0 else f"Refinement {index}")
        axes[0,index].axis("off")
    fig.tight_layout();fig.savefig(out/"refinement.png",dpi=150);plt.close(fig)
    for index,diag in enumerate(diagnostics):
        convert=lambda x:x.detach().cpu().numpy() if isinstance(x,torch.Tensor) else np.asarray(x)
        for key in ["txt_query_img_pos_cross_ref","sampling_locations","deformable_attention_weights","attention_weights"]:
            arrays[f"round_{index+1}_{key}"]=convert(diag[key])
        ref=convert(diag["txt_query_img_pos_cross_ref"])[0,:,begin,2,:]
        weights=convert(diag["attention_weights"])[begin,0]
        fig,ax=plt.subplots(figsize=(5,5))
        ax.imshow(arrays[f"stage_{index}_probability"],extent=(0,1,1,0),cmap="gray",vmin=0,vmax=1)
        dots=ax.scatter(ref[:,0],ref[:,1],c=weights,cmap="magma",s=25)
        fig.colorbar(dots,ax=ax,label="Cross-attention weight (head average)")
        ax.set(xlim=(0,1),ylim=(1,0),title=f"Round {index+1}: top-k, first phrase token")
        fig.tight_layout();fig.savefig(out/f"topk_round_{index+1}.png",dpi=150);plt.close(fig)
        locations=convert(diag["sampling_locations"])[0,:,begin]
        attention=convert(diag["deformable_attention_weights"])[0,:,begin]
        fig,axes=plt.subplots(1,4,figsize=(16,4))
        for level,ax in enumerate(axes):
            xy=locations[:,:,level,:,:].reshape(-1,2)
            weight=attention[:,:,level,:].reshape(-1)
            # Top 50 per level, not a reconstruction of the paper's exact selection.
            chosen=np.argsort(weight)[-min(50,len(weight)):]
            ax.imshow(arrays[f"stage_{index}_probability"],extent=(0,1,1,0),cmap="gray",vmin=0,vmax=1)
            ax.scatter(xy[chosen,0],xy[chosen,1],c=weight[chosen],cmap="magma",s=15)
            ax.set(xlim=(0,1),ylim=(1,0),title=f"Level {level+2}")
        fig.tight_layout();fig.savefig(out/f"sampling_round_{index+1}.png",dpi=150);plt.close(fig)
    np.savez_compressed(out/"diagnostics.npz",**arrays)
    (out/"analysis.json").write_text(json.dumps({"phrase_index":phrase,"token_interval":[begin,end],"attention_token":begin,"refinement_rounds":len(diagnostics),"coordinates":"normalized padded image geometry; samples outside [0,1] retained in NPZ","sampling_plot":"50 highest deformable weights per level across selected pixels, heads and offsets; first phrase token","paper_figure_reproduction":False},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ["config","checkpoint","sample","output"]:p.add_argument("--"+key,required=True)
    p.add_argument("--phrase",type=int,default=0);p.add_argument("--device",default="cpu")
    a=p.parse_args();cfg=read_config(a.config);torch.set_num_threads(cfg.get("cpu_threads",2))
    model=build_model(cfg,a.device);load_checkpoint(model,a.checkpoint);model.eval()
    sample=load_sample(a.sample,cfg,a.device)
    with torch.inference_mode():predictions,diagnostics=model(sample["fpn"],sample["language"],sample["noun_ids"])
    export_analysis(predictions,diagnostics,sample,a.output,a.phrase)

if __name__=="__main__":main()
