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
    # 图二的每个像素坐标加上它的光流即为该像素点对应在图一的坐标

    # scale grid to [-1,1]
    ##2019 code
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    # 取出光流v这个维度，原来范围是0~W-1，再除以W-1，范围是0~1，再乘以2，范围是0~2，再-1，范围是-1~1
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0  # 取出光流u这个维度，同上

    vgrid = vgrid.permute(0, 2, 3, 1)  # from B,2,H,W -> B,H,W,2，为什么要这么变呢？是因为要配合grid_sample这个函数的使用
    output = nn.functional.grid_sample(x, vgrid, mode='nearest', padding_mode='zeros',align_corners=True)
    #output=torch.grid_sampler(x, vgrid, 1, 0, align_corners=True)
    # mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
    # mask = nn.functional.grid_sample(mask, vgrid, align_corners=True)
    #
    # ##2019 author
    # mask[mask < 0.9999] = 0
    # mask[mask > 0] = 1

    ##2019 code
    # mask = torch.floor(torch.clamp(mask, 0 ,1))

    return output


class Warping(nn.Module):
    def __init__(self):
        super(Warping, self).__init__()

    def forward(self,x,flo):
        return warp(x,flo)



class TemporalNeckRAFT(nn.Module):
    def __init__(self):
        super(TemporalNeckRAFT, self).__init__()
        #输入1*256*80*80

        # feature network, context network, and update block
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
        # coords0 = coords_grid(N, H//8, W//8, device=img.device)
        # coords1 = coords_grid(N, H//8, W//8, device=img.device)

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
        # 输入1*256*80*80
        # fmap2映射到fmap1
        """ Estimate optical flow between pair of frames """
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        corr_fn = CorrBlock(fmap1, fmap2, radius=4)
        # run the context network
        # with autocast(enabled=self.args.mixed_precision):
        #     cnet = self.cnet(image1)
        #     # inp is the context feature
        #     net, inp = torch.split(cnet, [hdim, cdim], dim=1)
        #     net = torch.tanh(net)
        #     inp = torch.relu(inp)

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
            # with autocast(enabled=self.args.mixed_precision):
                # net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)
            net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # # upsample predictions(1,2,640,640)
            #  if up_mask is None:  # upsample?
            #      flow_up = upflow8(coords1 - coords0)
            #  else:
            #      flow_up = upsample_flow(coords1 - coords0, up_mask)

            # 不使用上采样(1,2,80,80)
            flow_up = coords1 - coords0
            flow_predictions.append(flow_up)

        # if test_mode:
        #     return coords1 - coords0, flow_up

        return flow_up

def compute_weight(flow, conv_feat):
    # 对channel维度进行L2Normalization
    flow_norm=torch.nn.functional.normalize(flow, p=2, dim=1)
    conv_norm=torch.nn.functional.normalize(conv_feat, p=2, dim=1)
    unsoftmax_weight=torch.sum(flow_norm*conv_norm,dim=1, keepdim=True)
    # print(weight.size())
    # weight:(1,1,80,80)
    # tile:weight:(1,1,80,80)-->(1,256,80,80)
    unsoftmax_weight = torch.tile(unsoftmax_weight, (1, 256, 1, 1))
    return unsoftmax_weight