import torch
from reshape import FittedWarp
import torch.nn as nn

class Net(torch.nn.Module):
    def __init__(self, w_shape):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(w_shape))

    def forward(self, x):
        return x @ self.w
    
    def input_size(self):
        return int(len(w))
    
class FittedWarpWithConvolution(torch.nn.Module):
    def __init__(self, w_shape):
        super().__init__()
        self.warp = FittedWarp(w_shape)
        self.conv1 = nn.TemporalConvolution(self.warp.input_shape[1],
                                                self.warp.input_shape[1],5)
    def forward(self,x):
        tmp = self.warp(x)
        tmp2 = self.conv1(tmp)
        return tmp2


