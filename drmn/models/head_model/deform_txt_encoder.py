import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch.nn.init import xavier_uniform_, constant_
from torch import optim
from drmn.ops.ms_deform_attn import MSDeformAttn

from .position_encoding import positionalencoding1d, positionalencoding2d


class TextDeformableTransformer(nn.Module):
    def __init__(self,
                 cfg,
                 d_model=256,
                 dropout=0.1,
                 n_heads=8, n_points=4, layer_num=4):
        super().__init__()
        self.max_sequence_length = cfg.max_sequence_length
        self.d_model = d_model
        self.layer_num = layer_num

        self.deformable_encoder_txt = nn.ModuleList(
                [TextDeformableTransformerLayer(cfg, d_model, dropout, n_heads, n_points) for i in range(self.layer_num)]
                )

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos


    @staticmethod
    def get_reference_points(batch, txt_spatial_shape, device):
        # for text
        L = txt_spatial_shape
        txt_reference_points_list = []
        for i in range(batch):
            ref_x = torch.linspace(0.5, L - 0.5, L, dtype=torch.float32, device=device)
            ref_x = ref_x.reshape(-1)[None, :, None] / L # 1, L, 1
            txt_reference_points_list.append(ref_x)
        txt_reference_points = torch.cat(txt_reference_points_list, 0)

        return txt_reference_points


    def forward(self, phrase, phrase_mask=None):
        # phrase sise
        B, L, E = phrase.shape
        assert E == self.d_model

        # position encoding
        pos_embed_txt = positionalencoding1d(B, d_model=E, max_len=L, device=phrase.device) ## B, L, E
        # text image encode
        txt_mask = ~phrase_mask # B, L

        # same modal ref point
        # txt_ref_points: B, L, 1
        txt_ref_points = self.get_reference_points(B, L, phrase.device)

        # prepare input
        out_txt = phrase
        ref_point = txt_ref_points
        all_shape = [B, E, L]
        mask = txt_mask

        for i in range(self.layer_num):
            # positional embedding
            out_txt = self.with_pos_embed(out_txt, pos_embed_txt)
            # cross attention
            out_txt = self.deformable_encoder_txt[i](out_txt, ref_point, all_shape, mask)

        return out_txt

class TextDeformableTransformerLayer(nn.Module):
    def __init__(self,
                 cfg,
                 d_model=256,
                 dropout=0.1,
                 n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn_txt = MyDeformAttnText(cfg, d_model, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_model)
        self.activation = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src


    def forward(self, src_point, ref_point, shapes, padding_mask=None):
        # decode param
        src = src_point
        txt_ref_points = ref_point
        B, C, L = shapes
        txt_mask = padding_mask

        # self attention
        # src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, shapes, padding_mask)
        src2 = self.self_attn_txt(src, txt_ref_points, src, [B,C,L], txt_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src

class MyDeformAttnText(nn.Module):
    def __init__(self, cfg, d_model=256, n_heads=8, n_points=4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_points = n_points
        # self.offset_range = 10/230
        self.max_sequence_length = cfg.max_sequence_length

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_points * 1)
        self.attention_weights = nn.Linear(d_model, n_heads * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = thetas.cos()[:, None]
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1).repeat(1, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_padding_mask=None):
        # shape
        B,C,L=input_spatial_shapes
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        # padding mask
        txt_padding_mask = input_padding_mask ## B, L

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(txt_padding_mask[..., None], float(0))
        value = value.view(B, Len_in, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(B, Len_q, self.n_heads, self.n_points, 1)
        attention_weights = self.attention_weights(query).view(B, Len_q, self.n_heads, self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(B, Len_q, self.n_heads, self.n_points)
        # N, Len_q, n_heads, n_points, 1
        sampling_locations = torch.clamp(reference_points[:, :, None, None, :] + sampling_offsets, 0, 1)

        output = self_attention1d(
            value, input_spatial_shapes, sampling_locations, attention_weights)
        output = self.output_proj(output)
        return output

def self_attention1d(value, input_spatial_shapes, sampling_locations, attention_weights):
    # need to use cuda version instead (?)
    B,C,L=input_spatial_shapes
    # batch size, number token, number head, head dims
    N_, S_, M_, D_ = value.shape
    # Lq_: number query, P_: sampling number采样点数
    _, Lq_, M_, P_, _ = sampling_locations.shape
    # [0, 1] -> [-1, 1] 因为要满足F.grid_sample的输入要求
    sampling_grids = 2 * sampling_locations - 1
    _value = value.permute(0,2,3,1).reshape(N_*M_, D_, L, 1)
    # N_, Lq_, M_, P_, 1 -> N_, M_, Lq_, P_, 1 -> N_*M_, Lq_, P_, 1 -> N_*M_, Lq_, P_, 2
    _sampling_grid = sampling_grids.permute(0,2,1,3,4).flatten(0, 1)
    _sampling_grid = torch.cat([_sampling_grid, torch.zeros_like(_sampling_grid, device=_sampling_grid.device)], dim=3)
    # N_*M_, D_, Lq_, P_
    # 用双线性插值从feature map上获取value，因为mask的原因越界所以要zeros的方法进行填充
    _sampling_value = F.grid_sample(_value, _sampling_grid,
                                        mode='bilinear', padding_mode='zeros', align_corners=False)
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_)
    _attention_weights = attention_weights.permute(0,2,1,3).reshape(N_*M_, 1, Lq_, P_)
    # 不同scale计算出的multi head attention 进行相加，返回output后还需要过一个Linear层
    output = (_sampling_value * _attention_weights).sum(-1).view(N_, M_*D_, Lq_)
    return output.transpose(1, 2).contiguous()
