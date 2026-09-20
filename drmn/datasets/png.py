"""PNG annotations to wordpiece-aligned masks, based on archived loader.

Uses first token of each phrase and exact bbox lookup as the archived loader.
Rejects inconsistent alignment rather than silently dropping annotations.
"""
import numpy as np
import torch
import torch.nn.functional as F


def annotation_targets(record,panoptic_annotation,categories,segment_ids,max_length,max_segments):
    raw=list(record["noun_vector"])
    # Truncating positive annotations would change evaluation coverage.
    if any(raw[max_length-2:]):raise ValueError("Caption truncates annotated noun tokens")
    raw=raw[:max_length-2]
    ids=np.array([0]+raw+[0]*(max_length-2-len(raw))+[0],dtype=np.int64)
    selected=ids[ids!=0]
    if not 0<len(selected)<=max_segments:raise ValueError("Invalid number of noun tokens")
    starts=np.r_[0,np.flatnonzero(selected[1:]!=selected[:-1])+1]
    intervals=np.r_[starts,len(selected)]
    valid_labels=[np.asarray(x) for x in record["labels"] if np.any(np.asarray(x)!=-2)]
    valid_boxes=[x for x in record["boxes"] if any(b!=[0,0,0,0] for b in x)]
    if len(valid_labels)!=len(selected) or len(valid_boxes)!=len(selected):
        raise ValueError("PNG labels/boxes/noun tokens do not align")
    masks=[];types=[];kinds=[]
    for start in starts:
        label=valid_labels[int(start)]
        types.append(1 if np.count_nonzero(label!=-2)==1 else 2)
        mask=np.zeros(segment_ids.shape,dtype=np.float32)
        kind=None
        for bbox in valid_boxes[int(start)]:
            if bbox==[0,0,0,0]:continue
            matches=[x for x in panoptic_annotation["segments_info"] if x["bbox"]==bbox]
            if len(matches)!=1:raise ValueError("Missing or ambiguous panoptic bounding box")
            segment=matches[0]
            mask += (segment_ids==segment["id"])
            kind=1 if categories[segment["category_id"]]["isthing"] else 2
        if not mask.any():raise ValueError("Annotated phrase has no matching segmentation pixels")
        masks.append(mask);kinds.append(kind)
    return {"noun_ids":ids,"phrase_intervals":intervals,"ann_types":np.asarray(types),"ann_categories":np.asarray(kinds),"original_masks":torch.from_numpy(np.stack(masks))}


def prepare_masks(original_masks,resized_hw):
    # Same ordering as archive: bilinear resize -> pad to 32 -> quarter -> >0.
    masks=F.interpolate(original_masks.unsqueeze(0),size=resized_hw,mode="bilinear",align_corners=False)
    h,w=resized_hw
    masks=F.pad(masks,(0,(-w)%32,0,(-h)%32))
    return (F.interpolate(masks,scale_factor=.25,mode="bilinear",align_corners=False)>0).float()[0]
