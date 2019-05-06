import torch
from torch import nn as nn
from torch.autograd import Variable
from torch.distributions import Gumbel
import numpy as np

from generative_playground.utils.gpu_utils import device


class SimplePolicy(nn.Module):
    '''
    Base class for a simple action selector
    '''
    def __init__(self):
        super().__init__()

    def forward(self, logits:Variable):
        '''
        Returns the index of the action chosen for each batch
        :param logits: batch x num_actions, float
        :return: ints, size = (batch)
        '''
        return NotImplementedError


class MaxPolicy(SimplePolicy):
    '''
    Returns one-hot encoding of the biggest logit value in each batch
    '''
    def forward(self, logits:Variable):
        _, max_ind = torch.max(logits,1) # argmax
        return max_ind


class SoftmaxRandomSamplePolicy(SimplePolicy):
    '''
    Randomly samples from the softmax of the logits
    # https://hips.seas.harvard.edu/blog/2013/04/06/the-gumbel-max-trick-for-discrete-distributions/
    TODO: should probably switch that to something more like
    http://pytorch.org/docs/master/distributions.html
    '''
    def __init__(self, bias=None):
        '''

        :param bias: a vector of log frequencies, to bias the sampling towards what we want
        '''
        super().__init__()
        self.gumbel = Gumbel(loc=0, scale=1)
        if bias is not None:
            self.bias = torch.from_numpy(np.array(bias)).to(dtype=torch.float32, device=device).unsqueeze(0)
        else:
            self.bias = None
    def forward(self, logits: Variable):
        '''

        :param logits: Logits to generate probabilities from, batch_size x out_dim float32
        :return:
        '''
        eff_logits = self.effective_logits(logits)
        x = self.gumbel.sample(logits.shape).to(device=device, dtype=eff_logits.dtype) + eff_logits
        _, out = torch.max(x, -1)
        return out

    def effective_logits(self, logits):
        if self.bias is not None:
            logits += self.bias
        return logits



class PolicyFromTarget(SimplePolicy):
    '''
    Just returns the next row from a target int sequence - useful for computing losses for encoders
    '''
    def __init__(self, target):
        super().__init__()
        self.target = target
        self.n = 0

    def forward(self, logits):
        if self.n < self.target.shape[1]:
            out = self.target[:,self.n]
            self.n += 1
            return out
        else:
            raise StopIteration

