"""Plot actual evaluation curves from one or more per-phrase evaluation JSON files."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results",nargs="+",required=True);p.add_argument("--labels",nargs="+")
    p.add_argument("--output",required=True);a=p.parse_args()
    labels=a.labels or [Path(f).stem for f in a.results]
    if len(labels)!=len(a.results):p.error("One label per result file is required")
    fig,axes=plt.subplots(1,5,figsize=(20,4));thresholds=np.linspace(0,1,1001)
    for filename,label in zip(a.results,labels):
        result=json.loads(Path(filename).read_text())["per_phrase"]
        iou=np.asarray(result["ious"]);types=np.asarray(result["ann_types"]);cats=np.asarray(result["ann_categories"])
        if iou.ndim!=1 or types.shape!=iou.shape or cats.shape!=iou.shape or not np.isfinite(iou).all() or np.any((iou<0)|(iou>1)):raise ValueError("Invalid per-phrase result")
        groups=[np.ones(len(iou),dtype=bool),types==1,types==2,cats==1,cats==2]
        for ax,mask,name in zip(axes,groups,["Overall","Singular","Plural","Things","Stuff"]):
            values=np.sort(iou[mask])
            if len(values):ax.plot(thresholds,(len(values)-np.searchsorted(values,thresholds,side="left"))/len(values),label=label)
            ax.set(xlabel="IoU threshold",ylabel="Recall",title=name,xlim=(0,1),ylim=(0,1))
    for ax in axes:
        if ax.lines:ax.legend()
    fig.tight_layout();out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);fig.savefig(out,dpi=180);plt.close(fig)

if __name__=="__main__":main()
