"""Encode PNG/COCO into validated frozen-feature samples using original weights."""
import _bootstrap
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from drmn.runtime import read_config,build_model,load_checkpoint
from drmn.models.frozen_encoders import FrozenImageEncoder,FrozenTextEncoder,read_resized_bgr
from drmn.datasets.png import annotation_targets,prepare_masks
from drmn.datasets.features import load_sample


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ["config","checkpoint","png-json","panoptic-json","panoptic-masks","images","output"]:p.add_argument("--"+key,required=True)
    p.add_argument("--bert-config",default="configs/bert/bert_config.json");p.add_argument("--vocab",default="configs/bert/vocab.txt")
    p.add_argument("--device",default="cpu");p.add_argument("--short-edge",type=int,default=800);p.add_argument("--max-size",type=int,default=1333);p.add_argument("--limit",type=int)
    a=p.parse_args();cfg=read_config(a.config);torch.set_num_threads(cfg.get("cpu_threads",2))
    # Strict integrity and model checks before processing a dataset.
    model=build_model(cfg);state=load_checkpoint(model,a.checkpoint);del model
    image=FrozenImageEncoder();image.load_original(state["fpn_model_state"]);image.to(a.device).eval()
    text=FrozenTextEncoder(a.bert_config,a.vocab,cfg["model"]["max_sequence_length"]);text.load_original(state["bert_model_state"]);text.to(a.device).eval();del state
    records=json.loads(Path(a.png_json).read_text());panoptic=json.loads(Path(a.panoptic_json).read_text())
    images={x["id"]:x for x in panoptic["images"]};anns={x["image_id"]:x for x in panoptic["annotations"]};categories={x["id"]:x for x in panoptic["categories"]}
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);written=[]
    for index,record in enumerate(records):
        if not any(v!=-2 for label in record["labels"] for v in label):continue
        if a.limit is not None and len(written)>=a.limit:break
        image_id=int(record["image_id"]);info=images[image_id];ann=anns[image_id]
        bgr,original_hw=read_resized_bgr(Path(a.images)/info["file_name"],a.short_edge,a.max_size)
        rgb=np.asarray(Image.open(Path(a.panoptic_masks)/ann["file_name"]).convert("RGB"),dtype=np.int64)
        ids=rgb[:,:,0]+256*rgb[:,:,1]+65536*rgb[:,:,2]
        if ids.shape!=original_hw:raise ValueError("Image and panoptic dimensions differ")
        target=annotation_targets(record,ann,categories,ids,cfg["model"]["max_sequence_length"],cfg["model"]["max_seg_num"])
        token_count=min(len(text.tokenizer.tokenize(record["caption"].strip())),cfg["model"]["max_sequence_length"]-2)
        if np.any(target["noun_ids"][token_count+1:]):raise ValueError("Annotated token exceeds tokenized caption")
        phrase=prepare_masks(target.pop("original_masks"),tuple(bgr.shape[-2:])).numpy()
        token=np.repeat(phrase,np.diff(target["phrase_intervals"]),axis=0)
        with torch.inference_mode():fpn=image(bgr);language=text(record["caption"])
        data={f"fpn_p{i+2}":v[0].cpu().numpy() for i,v in enumerate(fpn)}
        data.update(target,language=language[0].cpu().numpy(),targets_token=token,targets_phrase=phrase)
        path=out/f"{index:06d}_{image_id:012d}.npz"
        if path.exists():raise FileExistsError(path)
        np.savez_compressed(path,**data);load_sample(path,cfg);written.append(path.name)
    if not written:raise ValueError("No valid annotations encoded")
    manifest={"checkpoint":str(Path(a.checkpoint).resolve()),"png_json":str(Path(a.png_json).resolve()),"count":len(written),"files":written,"short_edge":a.short_edge,"max_size":a.max_size,"synthetic":False,"full_dataset":a.limit is None}
    (out/"SOURCE.json").write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))

if __name__=="__main__":main()
