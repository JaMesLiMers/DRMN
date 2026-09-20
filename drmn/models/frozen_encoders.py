"""Frozen encoder adapters, with explicit Detectron2-to-torchvision key mapping.

Backbone architecture follows archived Detectron2: R101, FrozenBN, stride in
conv1, BGR normalization, sum FPN. This adapter is reconstructed, not recovered.
"""
import re
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet101
from torchvision.ops.misc import FrozenBatchNorm2d
from PIL import Image
from drmn.models.bert_config import BertConfig
from drmn.models.encoder_bert import VisBert, set_visual_config, convert_sents_to_features
from drmn.models.tokenization import BertTokenizer


class DetectronFrozenBatchNorm2d(FrozenBatchNorm2d):
    """Match archived Detectron2 FrozenBN arithmetic, including inference path."""
    def __init__(self, num_features, eps=1e-5):
        super().__init__(num_features, eps)
        self.running_var.fill_(1.0-eps)

    def forward(self, x):
        if x.requires_grad:
            scale=self.weight*(self.running_var+self.eps).rsqrt()
            bias=self.bias-self.running_mean*scale
            return x*scale.reshape(1,-1,1,1).to(x.dtype)+bias.reshape(1,-1,1,1).to(x.dtype)
        return F.batch_norm(x,self.running_mean,self.running_var,self.weight,self.bias,training=False,eps=self.eps)


class FrozenImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.body=resnet101(weights=None,norm_layer=DetectronFrozenBatchNorm2d)
        self.body.fc=nn.Identity()
        # Detectron2 MSRA R101 applies downsampling in 1x1, torchvision in 3x3.
        for layer in (self.body.layer2,self.body.layer3,self.body.layer4):
            layer[0].conv1.stride=(2,2)
            layer[0].conv2.stride=(1,1)
        self.lateral=nn.ModuleList([nn.Conv2d(c,256,1) for c in [256,512,1024,2048]])
        self.output=nn.ModuleList([nn.Conv2d(256,256,3,padding=1) for _ in range(4)])
        self.register_buffer("pixel_mean",torch.tensor([103.53,116.28,123.675]).view(3,1,1))
        self.register_buffer("pixel_std",torch.ones(3,1,1))
        self.requires_grad_(False);self.eval()

    @staticmethod
    def source_key(key):
        if key.startswith(("pixel_mean","pixel_std")):return key
        if key.startswith(("lateral.","output.")):
            kind,index,tail=key.split(".",2)
            return f"backbone.fpn_{kind}{int(index)+2}.{tail}"
        key=key.removeprefix("body.")
        if key.startswith("conv1."):return "backbone.bottom_up.stem."+key
        if key.startswith("bn1."):return "backbone.bottom_up.stem.conv1.norm."+key[4:]
        match=re.match(r"layer([1-4])\.(\d+)\.(.*)",key)
        if not match:raise ValueError(f"Unknown backbone parameter {key}")
        level,block,tail=match.groups()
        tail=re.sub(r"^bn([123])\.",r"conv\1.norm.",tail)
        tail=tail.replace("downsample.0.","shortcut.").replace("downsample.1.","shortcut.norm.")
        return f"backbone.bottom_up.res{int(level)+1}.{block}.{tail}"

    def load_original(self,state):
        state={k.removeprefix("module."):v for k,v in state.items()}
        needed={k:self.source_key(k) for k in self.state_dict()}
        missing=set(needed.values())-state.keys()
        if missing:raise ValueError(f"Missing image encoder weights: {sorted(missing)[:8]}")
        used={k:state[v] for k,v in needed.items()}
        if not all(torch.isfinite(v).all() for v in used.values()):raise ValueError("Non-finite image weights")
        self.load_state_dict(used,strict=True)
        # Old ROI/semantic heads and fpn_wrapper are unused by archived forward().
        return {"loaded":len(used),"unused_non_backbone":len(state)-len(used)}

    @torch.no_grad()
    def forward(self,bgr):
        if bgr.ndim!=3 or bgr.shape[0]!=3:raise ValueError("Expected single BGR image [3,H,W]")
        x=(bgr.to(device=self.pixel_mean.device,dtype=torch.float32)-self.pixel_mean)/self.pixel_std
        h,w=x.shape[-2:];x=F.pad(x,(0,(-w)%32,0,(-h)%32)).unsqueeze(0)
        b=self.body;x=b.maxpool(b.relu(b.bn1(b.conv1(x))))
        features=[]
        for layer in (b.layer1,b.layer2,b.layer3,b.layer4):x=layer(x);features.append(x)
        last=self.lateral[3](features[3]);result=[self.output[3](last)]
        for i in [2,1,0]:
            last=self.lateral[i](features[i])+F.interpolate(last,scale_factor=2,mode="nearest")
            result.insert(0,self.output[i](last))
        return result


class FrozenTextEncoder(nn.Module):
    def __init__(self,config_path,vocab_path,max_length):
        super().__init__();set_visual_config()
        self.model=VisBert(BertConfig(str(config_path)))
        self.tokenizer=BertTokenizer(str(vocab_path),do_lower_case=True)
        self.max_length=max_length;self.requires_grad_(False);self.eval()

    def load_original(self,state):
        state={k.removeprefix("module."):v for k,v in state.items()}
        if not all(torch.isfinite(v).all() for v in state.values()):raise ValueError("Non-finite text weights")
        return self.load_state_dict(state,strict=True)

    @torch.no_grad()
    def forward(self,caption):
        features=convert_sents_to_features([caption],self.max_length,self.tokenizer)
        device=next(self.parameters()).device
        ids=torch.tensor([features[0].input_ids],device=device)
        segments=torch.tensor([features[0].segment_ids],device=device)
        mask=torch.tensor([features[0].input_mask],device=device)
        return self.model(ids,segments,mask)[0]


def read_resized_bgr(path,short_edge=800,max_size=1333):
    image=Image.open(path).convert("RGB");w,h=image.size
    scale=short_edge/min(h,w)
    if max(h,w)*scale>max_size:scale=max_size/max(h,w)
    new_h,new_w=int(h*scale+.5),int(w*scale+.5)
    image=image.resize((new_w,new_h),Image.Resampling.BILINEAR)
    return torch.from_numpy(np.asarray(image)[:,:,::-1].copy()).permute(2,0,1), (h,w)
