import torch
import torch.nn as nn
import torch.nn.functional as F

from .deform_img_encoder import DeformableTransformerEncoderLayer,DeformableTransformerEncoder
from .deform_txt_encoder import MyDeformAttnText, TextDeformableTransformer

from drmn.ops.ms_deform_attn import MSDeformAttn
from .position_encoding import positionalencoding2d, positionalencoding1d

class DeformEncoderHead(nn.Module):

    def __init__(self,
                 cfg,
                 in_channels=256,
                 dim_feedforward=256,
                 lang_channels=768,
                 nhead=8,
                 num_encoder_layers=3,
                 dropout=0.1,
                 activation="relu",
                 num_feature_levels=4,
                 enc_n_points=4,
                 ):
        super(DeformEncoderHead, self).__init__()
        # init param
        self.cfg=cfg
        self.in_channels = in_channels
        self.dim_feedforward = dim_feedforward
        self.lang_channels = lang_channels
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.dropout = dropout
        self.activation = activation
        self.num_feature_levels = num_feature_levels
        self.enc_n_points = enc_n_points

        # pre text encoder
        self.text_encoder_pre = nn.Sequential(
            nn.Linear(lang_channels, dim_feedforward),
        )

        self.img_encoder_pre = nn.ModuleList(
            [
                nn.Sequential(
                        nn.Conv2d(in_channels, dim_feedforward, kernel_size=4, padding=1, stride=2),
                        nn.GroupNorm(32, dim_feedforward),
                        nn.ReLU(inplace=True)
                    ), # for layer 1 H//2, W//2
                nn.Sequential(
                        nn.Conv2d(in_channels, dim_feedforward, kernel_size=1),
                        nn.GroupNorm(32, dim_feedforward),
                        nn.ReLU(inplace=True)
                    ), # for layer 2 H//2, W//2
                nn.Sequential(
                        nn.Conv2d(in_channels, dim_feedforward, kernel_size=1),
                        nn.GroupNorm(32, dim_feedforward),
                        nn.ReLU(inplace=True)
                    ), # for layer 3 H//4, W//4
                nn.Sequential(
                        nn.Conv2d(in_channels, dim_feedforward, kernel_size=1),
                        nn.GroupNorm(32, dim_feedforward),
                        nn.ReLU(inplace=True)
                    ), # for layer 4 H//8, W//8
            ]
        )

        # # init text deformable encoder
        # self.text_encoder = TextDeformableTransformer(cfg, dim_feedforward,
        #                                               dropout, nhead, enc_n_points, num_encoder_layers)

        # # init image deformable encoder
        # encoder_layer = DeformableTransformerEncoderLayer(dim_feedforward, dim_feedforward,
        #                                                   dropout, activation,
        #                                                   num_feature_levels, nhead, enc_n_points)
        # self.img_encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)
        # self.out_embed = nn.Parameter(torch.Tensor(num_feature_levels, dim_feedforward))

        # self._reset_parameters()

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
        # reduce dimention of text
        lang_feat = self.text_encoder_pre(lang_feat)
        # lang_feat = self.text_encoder(lang_feat, phrase_mask)

        # project dimention of image
        projected_imgs = []
        for l, img_feat in enumerate(img_feat_list):
            projected_imgs.append(self.img_encoder_pre[l](img_feat))

        # # original shape
        # img_size = []
        # for i in range(len(projected_imgs)):
        #     B, C, H, W = projected_imgs[i].shape
        #     img_size.append([B,C,H,W])

        # # convert fpn to padded version
        # srcs, masks = self.generate_mask(projected_imgs)

        # # embed posision info
        # pos_embeds = []
        # for src in srcs:
        #     # shape of feature map
        #     B, C, H, W = src.shape
        #     pos_embed_img = positionalencoding2d(B, d_model=C, height=H, width=W, device=src.device)
        #     pos_embeds.append(pos_embed_img)

        # # prepare input for encoder
        # src_flatten = []
        # mask_flatten = []
        # lvl_pos_embed_flatten = []
        # spatial_shapes = []

        # for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
        #     B, C, H, W = src.shape
        #     spatial_shape = (H, W)
        #     spatial_shapes.append(spatial_shape)
        #     src = src.flatten(2).transpose(1, 2)
        #     mask = mask.flatten(1)
        #     pos_embed = pos_embed.flatten(2).transpose(1, 2)
        #     lvl_pos_embed = pos_embed + self.out_embed[lvl].view(1, 1, -1)
        #     lvl_pos_embed_flatten.append(lvl_pos_embed)
        #     src_flatten.append(src)
        #     mask_flatten.append(mask)
        # src_flatten = torch.cat(src_flatten, 1)
        # mask_flatten = torch.cat(mask_flatten, 1)
        # lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        # spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        # level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        # valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # # encoder
        # memory = self.img_encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)

        # # prepare input for decoder
        # bs, _, c = memory.shape

        # # form back into feature map
        # memory_result = []
        # for i in range(len(level_start_index)):
        #     if i == len(level_start_index) - 1:
        #         level_result = memory[:, level_start_index[i]::, :]
        #     else:
        #         level_result = memory[:, level_start_index[i]: level_start_index[i+1], :]
        #     level_result = level_result.view([bs, spatial_shapes[i][0], spatial_shapes[i][1], c])
        #     original_h = img_size[i][2]
        #     original_w = img_size[i][3]
        #     level_result = level_result[:, :original_h, :original_w, :]
        #     memory_result.append(level_result.permute(0,3,1,2))

        memory_result=projected_imgs

        return memory_result, lang_feat


    def generate_mask(self, tensor_list):
        # tensor_list: [B, C, H, W]
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        mask_list = []
        for idx, img in enumerate(tensor_list):
            b, c, h, w = img.shape
            mask = torch.zeros((b, h, w), dtype=torch.bool, device=device)
            mask_list.append(mask)

        return tensor_list, mask_list


    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio




        # # init FPN
        # self.localization_fpn = SemanticFPNWrapper(in_channels=dim_feedforward, out_channels=dim_feedforward)

        # # post process
        # self.text_encoder_post = nn.Sequential(
        #     nn.Linear(dim_feedforward, out_channels),
        # )
        # self.img_encoder_post = nn.Sequential(
        #     nn.Conv2d(dim_feedforward, out_channels, kernel_size=(1, 1), stride=(1, 1), bias=False),
        #     nn.GroupNorm(32, 256, eps=1e-05, affine=True),
        #     nn.ReLU(inplace=True)
        # )
