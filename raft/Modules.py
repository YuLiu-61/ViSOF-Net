import numpy as np
import torch
import torch.nn as nn

from raft.update import BasicUpdateBlock
from raft.utils import coords_grid
from raft.corr import CorrBlock
import torch.nn.functional as F
from torch.autograd import Variable

def warp(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()
    x = x.cuda()
    grid=grid.cuda()
    vgrid = Variable(grid) + flo  # B,2,H,W
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
    vgrid = vgrid.permute(0, 2, 3, 1)  # from B,2,H,W -> B,H,W,2
    output = nn.functional.grid_sample(x, vgrid, mode='nearest', padding_mode='zeros',align_corners=True)
    return output


class TemporalNeckRAFT(nn.Module):
    def __init__(self):
        super(TemporalNeckRAFT, self).__init__()
        self.update_block = BasicUpdateBlock(hidden_dim=128)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape  # 1*256*80*80
        coords0 = coords_grid(N, H, W, device=img.device)
        coords1 = coords_grid(N, H, W, device=img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)

    def forward(self, fmap1, fmap2, iters=12, flow_init=None, upsample=True, test_mode=False):
        # input 1*256*80*80
        # fmap2-->fmap1
        """ Estimate optical flow between pair of frames """
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        corr_fn = CorrBlock(fmap1, fmap2, radius=4)
        net, inp = torch.split(fmap1, [128, 128], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(fmap1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)  # index correlation volume
            flow = coords1 - coords0
            net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            flow_up = coords1 - coords0
            flow_predictions.append(flow_up)


        return flow_up

def compute_weight(flow, conv_feat):
    # L2Normalization
    flow_norm=torch.nn.functional.normalize(flow, p=2, dim=1)
    conv_norm=torch.nn.functional.normalize(conv_feat, p=2, dim=1)
    unsoftmax_weight=torch.sum(flow_norm*conv_norm,dim=1, keepdim=True)
    # weight:(1,1,80,80)
    # tile:weight:(1,1,80,80)-->(1,256,80,80)
    unsoftmax_weight = torch.tile(unsoftmax_weight, (1, 256, 1, 1))
    return unsoftmax_weight