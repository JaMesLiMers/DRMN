import torch.nn as nn
import torch.nn.functional as F
from .kernel_head_new import ConvKernelHead
from .deform_iter_head import DeformIterHead
import torch
from .deform_encoder_head import DeformEncoderHead


class HeadNet(nn.Module):

    def __init__(
        self,
        cfg,
        num_stages=3,
        num_points=100
    ):
        super(HeadNet, self).__init__()

        self.cfg=cfg
        self.encoder_head = DeformEncoderHead(cfg=cfg)

        self.rpn_head = ConvKernelHead()
        self.roi_head = DeformIterHead(
            num_stages=num_stages,
            num_points=num_points
        )

    def forward(self, x, lang_feat, noun_vector_padding, phrase_mask=None, gts=None):
        x, lang_feat = self.encoder_head(x, lang_feat, phrase_mask)

        # choose lang valid feature
        lang_feat_valid = lang_feat.new_zeros((lang_feat.shape[0], \
            self.cfg.max_seg_num, lang_feat.shape[-1]))
        for i in range(len(lang_feat)):
            # # take average of target result
            # index_ref = noun_vector_padding[i][noun_vector_padding[i].nonzero().flatten()]
            # max_index = max(noun_vector_padding[i][noun_vector_padding[i].nonzero().flatten()]).item()
            # cur_lang_feat = lang_feat[i][noun_vector_padding[i].nonzero().flatten()]
            # cur_lang_feat_avg = torch.zeros_like(cur_lang_feat)
            # for j in range(1, max_index):
            #     selected_index = index_ref == j
            #     cur_lang_feat_avg[selected_index] = torch.mean(cur_lang_feat[selected_index],dim=0)
            # lang_feat_valid[i, :cur_lang_feat.shape[0], :] = cur_lang_feat_avg

            # original version
            cur_lang_feat = lang_feat[i][noun_vector_padding[i].nonzero().flatten()]
            lang_feat_valid[i, :cur_lang_feat.shape[0], :] = cur_lang_feat

        rpn_results = self.rpn_head(x, lang_feat_valid)

        (proposal_feats, x_feats, mlvl_feat, mask_preds) = rpn_results
        # proposal_feats: [B, 230, 256]
        # x_feats: [B, 256, H//8, W//8]
        # mask_preds: [B, 230, H//4, W//4]
        masks = [mask_preds]
        masks_iter, debug_results = self.roi_head.forward_train(x_feats, x, proposal_feats, mask_preds)
        masks = masks + masks_iter
        # debug_results = []
        return masks, debug_results
