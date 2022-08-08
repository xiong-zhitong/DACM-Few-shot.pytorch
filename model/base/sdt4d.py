r""" Implementation of center-pivot 4D convolution """

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, to_2tuple
import einops
from model.base.deformable_transformer import LocalAttention, ShiftWindowAttention, DAttentionBaseline, LayerNormProxy
import numpy as np
import cv2

class TransformerMLPWithConv(nn.Module):

    def __init__(self, channels, expansion, drop):
        
        super().__init__()
        
        self.dim1 = channels
        self.dim2 = channels * expansion
        self.linear1 = nn.Conv2d(self.dim1, self.dim2, 1, 1, 0)
        self.drop1 = nn.Dropout(drop)
        self.act = nn.GELU()
        self.linear2 = nn.Conv2d(self.dim2, self.dim1, 1, 1, 0) 
        self.drop2 = nn.Dropout(drop)
        self.dwc = nn.Conv2d(self.dim2, self.dim2, 3, 1, 1, groups=self.dim2)
    
    def forward(self, x):
        
        x = self.drop1(self.act(self.dwc(self.linear1(x))))
        x = self.drop2(self.linear2(x))
        
        return x

class TransformerStage(nn.Module):

    def __init__(self, fmap_size, window_size=3, ns_per_pt=4,
                 dim_in=64, dim_embed=128, depths=1, stage_spec='D', n_groups=4,
                 use_pe=False, sr_ratio=2,
                 heads=4, stride=1, offset_range_factor=3,
                 dwc_pe=False, no_off=False, fixed_pe=False,
                 attn_drop=0.1, proj_drop=0.1, expansion=4, drop=0.1, drop_path_rate=[0.1], use_dwc_mlp=True):

        super().__init__()
        fmap_size = to_2tuple(fmap_size)
        self.depths = depths
        hc = dim_embed // heads
        assert dim_embed == heads * hc
        self.proj = nn.Conv2d(dim_in, dim_embed, 1, 1, 0) if dim_in != dim_embed else nn.Identity()

        self.layer_norms = nn.ModuleList(
            [LayerNormProxy(dim_embed) for _ in range(2 * depths)]
        )
        self.mlps = nn.ModuleList(
            [
                TransformerMLPWithConv(dim_embed, expansion, drop)
                if use_dwc_mlp else TransformerMLP(dim_embed, expansion, drop)
                for _ in range(depths)
            ]
        )
        self.attns = nn.ModuleList()
        self.drop_path = nn.ModuleList()
        for i in range(depths):
            if stage_spec[i] == 'L':
                self.attns.append(
                    LocalAttention(dim_embed, heads, window_size, attn_drop, proj_drop)
                )
            elif stage_spec[i] == 'D':
                self.attns.append(
                    DAttentionBaseline(fmap_size, fmap_size, heads,
                    hc, n_groups, attn_drop, proj_drop,
                    stride, offset_range_factor, use_pe, dwc_pe,
                    no_off, fixed_pe)
                )
            elif stage_spec[i] == 'S':
                shift_size = math.ceil(window_size / 2)
                self.attns.append(
                    ShiftWindowAttention(dim_embed, heads, window_size, attn_drop, proj_drop, shift_size, fmap_size)
                )
            else:
                raise NotImplementedError(f'Spec: {stage_spec[i]} is not supported.')

            self.drop_path.append(DropPath(drop_path_rate[i]) if drop_path_rate[i] > 0.0 else nn.Identity())

    def forward(self, x):

        x = self.proj(x)

        positions = []
        references = []
        for d in range(self.depths):

            x0 = x
            x, pos, ref = self.attns[d](self.layer_norms[2 * d](x))
            x = self.drop_path[d](x) + x0
            x0 = x
            x = self.mlps[d](self.layer_norms[2 * d + 1](x))
            x = self.drop_path[d](x) + x0
            positions.append(pos)
            references.append(ref)

        return x, positions, references


class SDT4d(nn.Module):
    r""" CenterPivot 4D conv"""
    def __init__(self, fmap_size, in_channels, out_channels, stride, kernel_size, padding, shift=True):
        super(SDT4d, self).__init__()

        LD = 'D' if shift else 'L'
        self.conv1 = TransformerStage(fmap_size, window_size=3, ns_per_pt=4,
                 dim_in=in_channels, dim_embed=out_channels, depths=1, stage_spec=LD, n_groups=4,
                 use_pe=False, sr_ratio=2,
                 heads=4, stride=1, offset_range_factor=3)
        self.conv2 = TransformerStage(fmap_size, window_size=3, ns_per_pt=4,
                 dim_in=in_channels, dim_embed=out_channels, depths=1, stage_spec=LD, n_groups=4,
                 use_pe=False, sr_ratio=2,
                 heads=4, stride=1, offset_range_factor=3)

        self.stride34 = stride[2:]
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.idx_initialized = False

    def prune(self, ct):
        bsz, ch, ha, wa, hb, wb = ct.size()
        if not self.idx_initialized:
            idxh = torch.arange(start=0, end=hb, step=self.stride[2:][0], device=ct.device)
            idxw = torch.arange(start=0, end=wb, step=self.stride[2:][1], device=ct.device)
            self.len_h = len(idxh)
            self.len_w = len(idxw)
            self.idx = (idxw.repeat(self.len_h, 1) + idxh.repeat(self.len_w, 1).t() * wb).view(-1)
            self.idx_initialized = True
        ct_pruned = ct.view(bsz, ch, ha, wa, -1).index_select(4, self.idx).view(bsz, ch, ha, wa, self.len_h, self.len_w)

        return ct_pruned

    def forward(self, x):
        if self.stride[2:][-1] > 1:
            out1 = self.prune(x)
        else:
            out1 = x
        bsz, inch, ha, wa, hb, wb = out1.size()
        out1 = out1.permute(0, 4, 5, 1, 2, 3).contiguous().view(-1, inch, ha, wa)
        out1, pos, refs = self.conv1(out1)
        if ha == 50:
            pass
            #outf = out1.sum(dim=1) #N,h,w
            #outf = outf[0].detach().cpu().numpy() #
            #outf = (outf - outf.min())/(outf.max()-outf.min())
            #outf = (outf*255).astype(np.uint8)
            #outf = cv2.applyColorMap(outf, cv2.COLORMAP_JET)
            #cv2.imwrite('offsets/fmap1.jpg', outf)
            #null_img = np.ones([52,52])*255
            #pos = ((pos[0][0,0]+1)*25*8).abs().detach().cpu().numpy() #50,50,2
            #pos = pos.astype(np.int32).reshape([2500,2])
            #np.save('offsets/offset_q.npy', pos)
            
            #null_img[pos[:,0], pos[:,1]] = 0
            #cv2.imwrite('offsets/offfset.jpg',null_img.astype(np.uint8))
            #print(((pos[0][0,0]+1)*25).abs())  #N, group, 50, 50, 2
            #print((refs[0][0,0]+1)*25)
        outch, o_ha, o_wa = out1.size(-3), out1.size(-2), out1.size(-1)
        out1 = out1.view(bsz, hb, wb, outch, o_ha, o_wa).permute(0, 3, 4, 5, 1, 2).contiguous()

        bsz, inch, ha, wa, hb, wb = x.size()
        out2 = x.permute(0, 2, 3, 1, 4, 5).contiguous().view(-1, inch, hb, wb)
        out2, pos, refs = self.conv2(out2)
        if hb == 2:
            pass
            #pos = (pos[0][0,0]).abs().detach().cpu().numpy() #50,50,2
            #pos = pos.reshape([4,2]).astype(np.int32)
            #null_img = np.ones([3,3])*255
            #null_img[pos[:,0], pos[:,1]] = 0
            #cv2.imwrite('offsets/offfset2.jpg',null_img.astype(np.uint8))
            #np.save('offsets/offset_s.npy', pos)

        outch, o_hb, o_wb = out2.size(-3), out2.size(-2), out2.size(-1)
        out2 = out2.view(bsz, ha, wa, outch, o_hb, o_wb).permute(0, 3, 1, 2, 4, 5).contiguous()

        if out1.size()[-2:] != out2.size()[-2:] and self.padding[-2:] == (0, 0):
            out1 = out1.view(bsz, outch, o_ha, o_wa, -1).sum(dim=-1)
            out2 = out2.squeeze()

        y = out1 + out2
        return y



if __name__=='__main__':
    inp = torch.randn([1,64,56,56]).cuda()
    m1 = TransformerStage().cuda()
    tt,_,_ = m1(inp)
    print(tt.shape)
