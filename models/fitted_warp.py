import torch
from torch import nn as nn

from reshape import FittedWarp


class FittedWarpWithConvolution(torch.nn.Module):
    def __init__(self, w_shape=None, w=None):
        super().__init__()
        self.warp = FittedWarp(w_shape=w_shape, w=w)
        self.input_shape = self.warp.input_shape
        self.conv1 = nn.Conv2d(1,1,[5,1])
    def forward(self,x):
        tmp = self.warp(x)
        squeeze = False
        # do a convolution over time only, so don't mix dimensions
        # and inject a dummy channel dimension, and batch dim if necessary
        if len(x.shape)==2:
            tmp = torch.unsqueeze(tmp,0)
            squeeze = True
        tmp2 = torch.unsqueeze(tmp,1)
        #print(x.shape,tmp2.shape)
        tmp3 = self.conv1(tmp2)
        tmp3 = torch.squeeze(tmp3,1)
        if squeeze:
            tmp3 = torch.squeeze(tmp3,0)
        return tmp3


def conv_wrapper(x, conv):
    # takes either a 2-d tensor (time x other_dim) or a batch thereof,
    # and injects a channel and if necessary batch dimension to do conv2d
    squeeze = False
    if len(x.shape)==2:
        tmp = torch.unsqueeze(tmp,0)
        squeeze = True
    tmp2 = torch.unsqueeze(tmp,1)
    #print(x.shape,tmp2.shape)
    tmp3 = conv(tmp2)
    tmp4 = torch.squeeze(tmp3,1)
    if squeeze:
        tmp4 = torch.squeeze(tmp4,0)
    return tmp4