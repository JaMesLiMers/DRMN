"""Image + caption + WordPiece noun ids to phrase masks, using original weights."""
import _bootstrap
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from drmn.pretrained import load_encoders
from drmn.runtime import read_config,build_model,load_checkpoint,forward_sample
from drmn.models.frozen_encoders import FrozenImageEncoder,FrozenTextEncoder,read_resized_bgr
from drmn.evaluation.metrics import phrase_masks


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ["config","checkpoint","input","output"]:p.add_argument("--"+key,required=True)
    p.add_argument("--fpn-weights");p.add_argument("--bert-weights")
    p.add_argument("--bert-config",default="configs/bert/bert_config.json");p.add_argument("--vocab",default="configs/bert/vocab.txt")
    p.add_argument("--device",default="cpu");p.add_argument("--short-edge",type=int,default=800);p.add_argument("--max-size",type=int,default=1333)
    a=p.parse_args();cfg=read_config(a.config);torch.set_num_threads(cfg.get("cpu_threads",2))
    model=build_model(cfg,a.device);state=load_checkpoint(model,a.checkpoint);model.eval()
    image,text,encoder_source=load_encoders(a,cfg,state);del state
    entry=json.loads(Path(a.input).read_text());image_path=Path(entry["image"])
    if not image_path.is_absolute():image_path=Path(a.input).resolve().parent/image_path
    raw=entry["noun_ids"]
    if any(not isinstance(x,int) or x<0 for x in raw):raise ValueError("noun_ids must be nonnegative WordPiece phrase ids")
    if len(raw)!=len(text.tokenizer.tokenize(entry["caption"].strip())):raise ValueError("Provide one noun id per WordPiece token, without CLS/SEP")
    if len(raw)>cfg["model"]["max_sequence_length"]-2:raise ValueError("Caption too long")
    ids=[0]+raw+[0]*(cfg["model"]["max_sequence_length"]-len(raw)-1)
    selected=np.asarray(raw)[np.asarray(raw)!=0]
    if not len(selected):raise ValueError("No noun tokens")
    intervals=np.r_[0,np.flatnonzero(selected[1:]!=selected[:-1])+1,len(selected)]
    bgr,original_hw=read_resized_bgr(image_path,a.short_edge,a.max_size)
    with torch.inference_mode():
        sample={"fpn":image(bgr),"language":text(entry["caption"]),"noun_ids":torch.tensor([ids],device=a.device)}
        masks=phrase_masks(forward_sample(model,sample)[-1],intervals).cpu().numpy()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/"model_geometry_masks.npz",masks=masks)
    # Crop padding after upsampling, then restore original image size for viewing.
    h,w=bgr.shape[-2:];padded_h,padded_w=(h+31)//32*32,(w+31)//32*32
    for i,mask in enumerate(masks):
        view=Image.fromarray(mask.astype(np.uint8)*255).resize((padded_w,padded_h),Image.Resampling.NEAREST).crop((0,0,w,h)).resize((original_hw[1],original_hw[0]),Image.Resampling.NEAREST)
        view.save(out/f"phrase_{i:03d}.png")
    (out/"prediction.json").write_text(json.dumps({"caption":entry["caption"],"phrase_count":len(masks),"checkpoint":a.checkpoint,"model_geometry":list(masks.shape),"png_masks":"resized for visualization; evaluation uses model geometry"},indent=2))
    print(f"Saved {len(masks)} phrase masks to {out}")

if __name__=="__main__":main()
