import torch
from reshape import WarpMatrix

class Net(torch.nn.Module):
    def __init__(self, w_shape):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(w_shape))

    def forward(self, x):
        return x @ self.w
    
    def input_size(self):
        return int(len(w))
    
class FittedWarp(torch.nn.Module):
    def __init__(self, w_shape):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(w_shape))

    def forward(self, x):
        tmp1 = x @ self.w
        tmp = torch.nn.Sigmoid()(tmp1)
        trans_mat = WarpMatrix.apply(tmp)
        return trans_mat @ x


