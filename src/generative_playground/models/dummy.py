import torch.distributions as tdist
import torch.nn as nn
import torch

class DummyModel(nn.Module):
    def __init__(self, output_shape, random=True):
        super().__init__()
        if random:
            n = tdist.Normal(0,1)
            self.data = n.sample(output_shape)
        else:
            self.data = torch.ones(*output_shape)
        self.output_shape = output_shape

    def forward(self, *input):
        return self.data

class PassthroughModel(nn.Module):
    def __init__(self, output_shape):
        super().__init__()
        self.output_shape = [None, *(output_shape[1:])]

    def forward(self, input):
        assert list(input.size()[1:]) == self.output_shape[1:]
        return input