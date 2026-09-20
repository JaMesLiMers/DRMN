import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .multiheadattention import (MultiheadAtten, Ffn)
from .position_encoding import positionalencoding2d, positionalencoding1d
from .deform_txt_decoder import DeformableTransformerDecoder

class KernelUpdateHead(nn.Module):

    def __init__(self,
                 num_heads=8,
                 num_mask_fcs=3,
                 in_channels=256,
                 out_channels=256,
                 dropout=0.0,
                 with_ffn=True,
                 mask_transform_stride=2,
                 num_points=100
                 ):
        super(KernelUpdateHead, self).__init__()


        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout

        self.num_heads = num_heads
        self.with_ffn = with_ffn
        self.mask_transform_stride = mask_transform_stride
        self.num_points = num_points

        self.loc_convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.GroupNorm(32, 256, eps=1e-05, affine=True),
            nn.ReLU(inplace=True)
        )


        self.text_decoder = DeformableTransformerDecoder(n_sample=num_points)

        self.topk_attn = MultiheadAtten(in_channels, num_heads, dropout)
        self.topk_norm = nn.LayerNorm(in_channels, \
                eps=1e-05, elementwise_affine=True)

        if self.with_ffn:
            self.ffn = Ffn()
            self.ffn_norm = nn.LayerNorm(in_channels, eps=1e-05, elementwise_affine=True)
            self.ffn_pre = Ffn()
            self.ffn_norm_pre = nn.LayerNorm(in_channels, eps=1e-05, elementwise_affine=True)

        self.mask_fcs = nn.ModuleList()
        for _ in range(num_mask_fcs):
            self.mask_fcs.append(
                nn.Linear(in_channels, in_channels, bias=False))
            self.mask_fcs.append(
                nn.LayerNorm((256,), eps=1e-05, elementwise_affine=True))
            self.mask_fcs.append(nn.ReLU(inplace=True))

        self.fc_mask = nn.Linear(in_channels, out_channels)


    def forward(self, x, mlvl_feats, proposal_feat, mask_preds, mask_shape=None):
        K = self.num_points
        x = self.loc_convs(x)
        # proposal_feat: [B, 230, 256]
        B, N = proposal_feat.shape[:2]
        # x: [B, 256, H//8, W//8] <--> Features $F$
        C, H, W = x.shape[-3:]
        # mask_preds: [B, 230, H//4, W//4] <--> $M$
        mask_h, mask_w = mask_preds.shape[-2:]
        # if mask_h != H or mask_w != W:
        #     gather_mask = F.interpolate(
        #         mask_preds, (H, W), align_corners=False, mode='bilinear')
        #     # gather_mask: [B, 230, H//8, W//8]
        # else:
        gather_mask = mask_preds

        # debug results:
        # txt_query_img_pos_cross_ref (B x K x L x n_levels x 2)
        # sampling_locations: B, K x L, n_heads, n_levels, n_points, 2
        # attention_weights: B, K x L, n_heads, n_levels, n_points
        topk_feats, debug_results = self.text_decoder(mlvl_feats, proposal_feat, gather_mask, level_index=2)
        # [B, K, N, C]
        topk_feats = topk_feats.permute(0,2,1,3)
        # [B, N, K, C]
        obj_feat = proposal_feat.unsqueeze(2)
        # [B, N, 1, C]

        topk_feats = topk_feats.reshape(B*N, K, C)
        obj_feat = obj_feat.reshape(B*N, 1, C)
        topk_feats = topk_feats.transpose(0, 1)
        # [B*N, K, C]
        obj_feat = obj_feat.transpose(0, 1)
        # [B*N, 1, C]

        # [B, N, K]
        obj_feat, attention_weights = self.topk_attn(obj_feat, topk_feats, debug=True)
        obj_feat = self.topk_norm(obj_feat)

        obj_feat = obj_feat.transpose(0, 1)
        obj_feat = obj_feat.reshape(B, N, 1, C).squeeze(2)
        obj_feat = self.ffn_norm_pre(self.ffn_pre(obj_feat))
        # [B, N, C]

        mask_feat = obj_feat

        for reg_layer in self.mask_fcs:
            mask_feat = reg_layer(mask_feat)
        mask_feat = self.fc_mask(mask_feat)
        # [B, N, C, K*K] -> [B*N, C, K, K]


        mask_x = x
        # new_mask_preds: [B, C, H//8, W//8]
        new_mask_preds = torch.einsum('bchw,bnc->bnhw', mask_x, mask_feat)

        if self.mask_transform_stride == 2:
            new_mask_preds = F.interpolate(
                new_mask_preds,
                scale_factor=2,
                mode='bilinear',
                align_corners=False)

        debug_results["attention_weights"] = attention_weights.detach().cpu().numpy()
        debug_results["mask_x"] = mask_x.detach().cpu().numpy()
        debug_results["mask_feat"] = mask_feat.detach().cpu().numpy()
        return new_mask_preds, obj_feat, debug_results



    @staticmethod
    def get_reference_points(img_spatial_shapes, txt_spatial_shape, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(img_spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]

        # for text
        L = txt_spatial_shape[1]
        ref_x = torch.linspace(0.5, L - 0.5, L, dtype=torch.float32, device=device)
        ref_x = ref_x.reshape(-1)[None, :, None] / L # 1, L, 1
        txt_reference_points = ref_x.repeat(txt_spatial_shape[0], 1, 1)

        return reference_points, txt_reference_points, ref
