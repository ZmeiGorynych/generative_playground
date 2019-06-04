import torch
from torch import nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Gumbel
import numpy as np
import random

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
    def __init__(self, temperature=1.0, eps=0.0):
        '''

        :param bias: a vector of log frequencies, to bias the sampling towards what we want
        '''
        super().__init__()
        self.gumbel = Gumbel(loc=0, scale=1)
        self.temperature = torch.tensor(temperature, requires_grad=False)
        self.eps = eps

    def set_temperature(self, new_temperature):
        if self.temperature != new_temperature:
            self.temperature = torch.tensor(new_temperature,
                                            dtype=self.temperature.dtype,
                                            device=self.temperature.device)

    def forward(self, logits: Variable, priors=None):
        '''

        :param logits: Logits to generate probabilities from, batch_size x out_dim float32
        :return:
        '''
        if random.random() < self.eps:
            if priors is None:
                new_logits = torch.zeros_like(logits)
                new_logits[logits < logits.max()-1000] = -1e4
            else:
                new_logits = priors
        else:
            new_logits = logits

        if self.temperature.dtype != logits.dtype or self.temperature.device != logits.device:
            self.temperature = torch.tensor(self.temperature,
                                            dtype=logits.dtype,
                                            device=logits.device,
                                            requires_grad=False)


        # temperature is applied to model only, not priors!
        if priors is None:
            new_logits_t = new_logits/self.temperature
        else:
            new_logits_t = ((new_logits-priors) / self.temperature) + priors

        eff_logits = self.effective_logits(new_logits_t)
        x = self.gumbel.sample(logits.shape).to(device=device, dtype=eff_logits.dtype) + eff_logits
        _, out = torch.max(x, -1)
        all_logp = F.log_softmax(eff_logits, dim=1)
        self.logp = torch.cat([this_logp[this_ind:(this_ind+1)] for this_logp, this_ind in zip(all_logp, out)])
        return out

    def effective_logits(self, logits):
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

