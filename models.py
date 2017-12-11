import torch

class Net(torch.nn.Module):
    def __init__(self, w_shape):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(w_shape))

    def forward(self, x):
        return x @ self.w

