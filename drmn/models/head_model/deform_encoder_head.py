# decompyle3 version 3.9.3
# Python bytecode version base 3.7.0 (3394)
# Decompiled from: Python 3.10.12 (main, Aug 31 2026, 10:18:17) [GCC 11.4.0]
# Recovered from original Python bytecode; see docs/consistency-audit.md.
# Compiled at: 2023-05-31 12:48:04
# Size of source mod 2**32: 8131 bytes
import torch
import torch.nn as nn
import torch.nn.functional as F
from .deform_img_encoder import DeformableTransformerEncoderLayer, DeformableTransformerEncoder
from .deform_txt_encoder import MyDeformAttnText, TextDeformableTransformer
from drmn.ops.ms_deform_attn import MSDeformAttn
from .position_encoding import positionalencoding2d, positionalencoding1d

class DeformEncoderHead(nn.Module):

    def __init__(self, cfg, in_channels=256, dim_feedforward=256, lang_channels=768, nhead=8, num_encoder_layers=2, dropout=0.1, activation='relu', num_feature_levels=4, enc_n_points=4):
        super(DeformEncoderHead, self).__init__()
        self.cfg = cfg
        self.in_channels = in_channels
        self.dim_feedforward = dim_feedforward
        self.lang_channels = lang_channels
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.dropout = dropout
        self.activation = activation
        self.num_feature_levels = num_feature_levels
        self.enc_n_points = enc_n_points
        self.text_encoder_pre = nn.Sequential(nn.Linear(lang_channels, dim_feedforward))
        self.img_encoder_pre = nn.ModuleList([
         nn.Sequential(nn.Conv2d(in_channels, dim_feedforward, kernel_size=4, padding=1, stride=2), nn.GroupNorm(32, dim_feedforward), nn.ReLU(inplace=True)),
         nn.Sequential(nn.Conv2d(in_channels, dim_feedforward, kernel_size=1), nn.GroupNorm(32, dim_feedforward), nn.ReLU(inplace=True)),
         nn.Sequential(nn.Conv2d(in_channels, dim_feedforward, kernel_size=1), nn.GroupNorm(32, dim_feedforward), nn.ReLU(inplace=True)),
         nn.Sequential(nn.Conv2d(in_channels, dim_feedforward, kernel_size=1), nn.GroupNorm(32, dim_feedforward), nn.ReLU(inplace=True))])
        self.text_encoder = TextDeformableTransformer(cfg, dim_feedforward, dropout, nhead, enc_n_points, num_encoder_layers)
        encoder_layer = DeformableTransformerEncoderLayer(dim_feedforward, dim_feedforward, dropout, activation, num_feature_levels, nhead, enc_n_points)
        self.img_encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)
        self.out_embed = nn.Parameter(torch.Tensor(num_feature_levels, dim_feedforward))
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
            if isinstance(m, MyDeformAttnText):
                m._reset_parameters()

        nn.init.normal_(self.out_embed)

    def forward(self, img_feat_list, lang_feat, phrase_mask=None):
        lang_feat = self.text_encoder_pre(lang_feat)
        projected_imgs = []
        for l, img_feat in enumerate(img_feat_list):
            projected_imgs.append(self.img_encoder_pre[l](img_feat))

        img_size = []
        for i in range(len(projected_imgs)):
            B, C, H, W = projected_imgs[i].shape
            img_size.append([B, C, H, W])

        srcs, masks = self.generate_mask(projected_imgs)
        pos_embeds = []
        for src in srcs:
            B, C, H, W = src.shape
            pos_embed_img = positionalencoding2d(B, d_model=C, height=H, width=W, device=src.device)
            pos_embeds.append(pos_embed_img)

        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            B, C, H, W = src.shape
            spatial_shape = (H, W)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.out_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)

        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=(torch.long), device=(src_flatten.device))
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
        memory = self.img_encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)
        bs, _, c = memory.shape
        memory_result = []
        for i in range(len(level_start_index)):
            if i == len(level_start_index) - 1:
                level_result = memory[:, level_start_index[i]:, :]
            else:
                level_result = memory[:, level_start_index[i]:level_start_index[i + 1], :]
            level_result = level_result.view([bs, spatial_shapes[i][0], spatial_shapes[i][1], c])
            original_h = img_size[i][2]
            original_w = img_size[i][3]
            level_result = level_result[:, :original_h, :original_w, :]
            memory_result.append(level_result.permute(0, 3, 1, 2))

        return (
         memory_result, lang_feat)

    def generate_mask(self, tensor_list):
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        mask_list = []
        for (idx, img) in enumerate(tensor_list):
            (b, c, h, w) = img.shape
            mask = torch.zeros((b, h, w), dtype=(torch.bool), device=device)
            mask_list.append(mask)

        return (
         tensor_list, mask_list)

    def get_valid_ratio(self, mask):
        (_, H, W) = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio
