# decompyle3 version 3.9.3
# Python bytecode version base 3.7.0 (3394)
# Decompiled from: Python 3.10.12 (main, Aug 31 2026, 10:18:17) [GCC 11.4.0]
# Recovered from original Python bytecode; see docs/最终一致性核对.md.
# Compiled at: 2023-07-21 14:04:45
# Size of source mod 2**32: 13371 bytes
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from drmn.ops.ms_deform_attn import MSDeformAttn
from .position_encoding import positionalencoding2d, positionalencoding1d

class DeformableTransformerDecoderLayer(nn.Module):

    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, activation='relu', n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        if pos is None:
            return tensor
        return tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, original_tgt, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask=None):
        (tgt2, debug_result) = self.cross_attn((self.with_pos_embed(tgt, query_pos)), reference_points,
          src,
          src_spatial_shapes, level_start_index, src_padding_mask, debug=True)
        tgt = original_tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt = self.forward_ffn(tgt)
        return (
         tgt, debug_result)


class DeformableTransformerDecoder(nn.Module):

    def __init__(self, d_model=256, d_ffn=256, dropout=0.1, activation='relu', n_levels=4, n_heads=8, n_points=4, n_sample=5, num_feature_levels=4):
        super().__init__()
        self.cross_layer = DeformableTransformerDecoderLayer(d_model, d_ffn, dropout, activation, n_levels, n_heads, n_points)
        self.n_sample = n_sample
        self.n_heads = n_heads
        self.n_points = n_points
        self.out_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()

        nn.init.normal_(self.out_embed)

    def generate_cross_ref(self, img_src, spatial_shapes, level_start_index, txt_src, img_ref_points, mode='sim_top_k', pred_mask=None, start_index=0):
        (B, HW, level, _) = img_ref_points.shape
        (B, all_HW, C) = img_src.shape
        if mode == "sim_top_k":
            all_ref_point = []
            sim_txt_query_image = torch.bmm(txt_src, img_src.permute(0, 2, 1))
            (value_txt_query_img, index_txt_query_img) = torch.topk(sim_txt_query_image, k=(self.n_sample), dim=2)
            for i in range(index_txt_query_img.shape[2]):
                index_txt_query_img_i = index_txt_query_img[:, :, i]
                txt_query_img_pos_cross_ref = img_ref_points[(torch.arange(B)[:, None], index_txt_query_img_i)]
                all_ref_point.append(txt_query_img_pos_cross_ref)

            all_ref_point = torch.stack(all_ref_point, 1)
            (B, K, L, _, _) = all_ref_point.shape
            all_ref_point = all_ref_point.view(B, K * L, 4, 2)
        if mode == "according_to_pred_mask":
            assert pred_mask is not None, "pred_mask must have value"
            (H, W) = spatial_shapes[start_index]
            pred_mask = F.interpolate(pred_mask,
              (H, W), align_corners=False, mode="bilinear")
            (value_txt_query_img, index_txt_query_img) = torch.topk((pred_mask.flatten(-2)), k=(self.n_sample))
            index_txt_query_img = level_start_index[start_index] + index_txt_query_img
            all_ref_point = img_ref_points[(torch.arange(B)[:, None, None], index_txt_query_img)].permute(0, 2, 1, 3, 4)
            (B, K, L, _, _) = all_ref_point.shape
            all_ref_point = all_ref_point.reshape(B, K * L, 4, 2)
            txt_query_img_pos_cross_feature = img_src[(torch.arange(B)[:, None, None], index_txt_query_img)].permute(0, 2, 1, 3).reshape(B, K * L, C)
        else:
            raise NotImplementedError
        return (all_ref_point, txt_query_img_pos_cross_feature)

    @staticmethod
    def get_reference_points(img_spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for (lvl, (H_, W_)) in enumerate(img_spatial_shapes):
            (ref_y, ref_x) = torch.meshgrid(torch.linspace(0.5, (H_ - 0.5), H_, dtype=(torch.float32), device=device), torch.linspace(0.5, (W_ - 0.5), W_, dtype=(torch.float32), device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)

        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def prepare_input(self, mlvl_feats, lang_feat):
        projected_img = mlvl_feats
        img_size = []
        for i in range(len(projected_img)):
            (B, C, H, W) = projected_img[i].shape
            img_size.append([B, C, H, W])

        (srcs, masks) = self.generate_mask(projected_img)
        pos_embeds = []
        for src in srcs:
            (B, C, H, W) = src.shape
            pos_embed_img = positionalencoding2d(B, d_model=C, height=H, width=W, device=src.device)
            pos_embeds.append(pos_embed_img)

        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for (lvl, (src, mask, pos_embed)) in enumerate(zip(srcs, masks, pos_embeds)):
            (B, C, H, W) = src.shape
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
        (B, C, L) = lang_feat.shape
        lang_pos_embedding = positionalencoding1d(B, d_model=C, max_len=L, device=lang_feat.device)
        lang_pos_embedding = lang_pos_embedding.permute(0, 2, 1)
        return (
         src_flatten, lang_feat, spatial_shapes, level_start_index, valid_ratios, lang_pos_embedding, mask_flatten)

    def forward(self, mlvl_feats, lang_feat, pred_mask, level_index=0):
        (src_flatten, lang_feat, spatial_shapes, level_start_index, valid_ratios, lang_pos_embedding, mask_flatten) = self.prepare_input(mlvl_feats, lang_feat)
        (B, L, C) = lang_feat.shape
        (H, W) = spatial_shapes[0]
        img_ref_points = self.get_reference_points(spatial_shapes, valid_ratios, src_flatten.device)
        (txt_query_img_pos_cross_ref, txt_query_img_pos_cross_feature) = self.generate_cross_ref(src_flatten, spatial_shapes, level_start_index, lang_feat, img_ref_points, mode="according_to_pred_mask", pred_mask=pred_mask, start_index=level_index)
        lang_feat_repeat = lang_feat[:, None, :, :].repeat(1, self.n_sample, 1, 1).view(B, self.n_sample * L, C)
        lang_pos_embedding_repeat = lang_pos_embedding[:, None, :, :].repeat(1, self.n_sample, 1, 1).view(B, self.n_sample * L, C)
        (output_lang, (sampling_locations, attention_weights)) = self.cross_layer(lang_feat_repeat, txt_query_img_pos_cross_feature, lang_pos_embedding_repeat, txt_query_img_pos_cross_ref, src_flatten, spatial_shapes, level_start_index, mask_flatten)
        output_lang = output_lang.view(B, self.n_sample, L, C)
        debug_result = {'txt_query_img_pos_cross_ref':(txt_query_img_pos_cross_ref.view)(B, self.n_sample, L, 4, 2),
         'sampling_locations':(sampling_locations.view)(B, self.n_sample, L, self.n_heads, 4, self.n_points, 2),
         'attention_weights':(attention_weights.view)(B, self.n_sample, L, self.n_heads, 4, self.n_points)}
        return (
         output_lang, debug_result)

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


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
