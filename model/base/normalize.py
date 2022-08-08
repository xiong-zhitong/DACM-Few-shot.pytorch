r""" Provides functions that builds/manipulates correlation tensors """
import torch
import pdb
import torch.nn.functional as F


class Normalize:

    @classmethod
    def multilayer_norm(cls, query_feats, support_feats, stack_ids):
        eps = 1e-5

        normed_s_feats = []
        normed_q_feats = []

        for idx, (query_feat, support_feat) in enumerate(zip(query_feats, support_feats)):
            bsz, ch, hb, wb = support_feat.size()
            support_feat = support_feat.view(bsz, ch, -1)
            support_feat = support_feat / (support_feat.norm(dim=1, p=2, keepdim=True) + eps)
            support_feat = support_feat.permute([0,2,1])
            normed_s_feats.append(support_feat)

            bsz, ch, ha, wa = query_feat.size()
            query_feat = query_feat.view(bsz, ch, -1)
            query_feat = query_feat / (query_feat.norm(dim=1, p=2, keepdim=True) + eps)
            query_feat = query_feat.permute([0,2,1])
            normed_q_feats.append(query_feat)


        ns_l4 = torch.stack(normed_s_feats[-stack_ids[0]:]).mean(0)
        nq_l4 = torch.stack(normed_q_feats[-stack_ids[0]:]).mean(0)
        ns_l3 = torch.stack(normed_s_feats[-stack_ids[1]:-stack_ids[0]]).mean(0)
        nq_l3 = torch.stack(normed_q_feats[-stack_ids[1]:-stack_ids[0]]).mean(0)
        ns_l2 = torch.stack(normed_s_feats[-stack_ids[2]:-stack_ids[1]]).mean(0)
        nq_l2 = torch.stack(normed_q_feats[-stack_ids[2]:-stack_ids[1]]).mean(0)

        return [ns_l2,ns_l3,ns_l4], [nq_l2,nq_l3,nq_l4]
    
    
    @classmethod
    def multilayer_correlation(cls, query_feats, support_feats, stack_ids):
        eps = 1e-5

        corrs = []
        for idx, (query_feat, support_feat) in enumerate(zip(query_feats, support_feats)):
            bsz, ch, hb, wb = support_feat.size()
            support_feat = support_feat.view(bsz, ch, -1)
            support_feat = support_feat / (support_feat.norm(dim=1, p=2, keepdim=True) + eps)

            bsz, ch, ha, wa = query_feat.size()
            query_feat = query_feat.view(bsz, ch, -1)
            query_feat = query_feat / (query_feat.norm(dim=1, p=2, keepdim=True) + eps)

        corr_l4 = torch.stack(corrs[-stack_ids[0]:]).transpose(0, 1).contiguous()
        corr_l3 = torch.stack(corrs[-stack_ids[1]:-stack_ids[0]]).transpose(0, 1).contiguous()
        corr_l2 = torch.stack(corrs[-stack_ids[2]:-stack_ids[1]]).transpose(0, 1).contiguous()

        return [corr_l4, corr_l3, corr_l2]
    
    
    @classmethod
    def multiscale_masks(cls,support_mask):
        N, H, W = support_mask.shape
        H1 = H//8; W1 = W//8
        H2 = H//16;W2 = W//16
        H3 = H//32;W3 = W//32
        s_mask1 = F.interpolate(support_mask.view(N,1,H,W), size=(H1,W1), mode='nearest',recompute_scale_factor=False)
        s_mask2 = F.interpolate(support_mask.view(N,1,H,W), size=(H2,W2), mode='nearest',recompute_scale_factor=False)
        s_mask3 = F.interpolate(support_mask.view(N,1,H,W), size=(H3,W3), mode='nearest',recompute_scale_factor=False)
        return [s_mask1.view(N,H1*W1), s_mask2.view(N,H2*W2), s_mask3.view(N,H3*W3)]


