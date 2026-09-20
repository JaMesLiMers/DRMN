import torch
import torch.nn as nn
import torch.nn.functional as F

from .kernel_update_head import KernelUpdateHead
# from deform_txt_decoder import DeformableTransformerDecoder,DeformableTransformerDecoderLayer

from .position_encoding import positionalencoding2d, positionalencoding1d

class DeformIterHead(nn.Module):

    def __init__(self,
        num_stages=3,
        num_points=100,
    ):
        super(DeformIterHead, self).__init__()
        self.num_stages = num_stages
        self.mask_head = nn.ModuleList()
        for i in range(num_stages):
            self.mask_head.append(KernelUpdateHead(num_points=num_points))


    def _mask_forward(self, stage, x, mlvl_feats, kernels, mask_preds):
        mask_head = self.mask_head[stage]
        mask_preds, kernels, debug_results = mask_head(
            x, mlvl_feats, kernels, mask_preds)

        return mask_preds, kernels, debug_results


    def forward_train(self, x, mlvl_feats, proposal_feats, mask_preds):
        # object_feats = proposal_feats
        kernels = proposal_feats
        all_stage_mask_results = []
        all_stage_debug_results = []
        for stage in range(self.num_stages):
            mask_preds, kernels, debug_results = self._mask_forward(stage, x, mlvl_feats, kernels,
                                              mask_preds)
            all_stage_mask_results.append(mask_preds)
            all_stage_debug_results.append(debug_results)

        return all_stage_mask_results, all_stage_debug_results
