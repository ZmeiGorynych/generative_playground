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
    def __init__(self, temperature=1.0, eps=0.0, do_entropy=True):
        '''

        :param bias: a vector of log frequencies, to bias the sampling towards what we want
        '''
        super().__init__()
        self.gumbel = Gumbel(loc=0, scale=1)
        self.temperature = torch.tensor(temperature, requires_grad=False)
        self.eps = eps
        self.do_entropy = do_entropy
        if self.do_entropy:
            self.entropy = None

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
        # epsilon-greediness
        if random.random() < self.eps:
            if priors is None:
                new_logits = torch.zeros_like(logits)
                new_logits[logits < logits.max()-1000] = -1e4
            else:
                new_logits = priors
            if self.do_entropy:
                self.entropy = torch.zeros_like(logits).sum(dim=1)
        else:
            if self.temperature.dtype != logits.dtype or self.temperature.device != logits.device:
                self.temperature = torch.tensor(self.temperature,
                                                dtype=logits.dtype,
                                                device=logits.device,
                                                requires_grad=False)
            # print('temperature: ', self.temperature)


            # temperature is applied to model only, not priors!
            if priors is None:
                new_logits = logits/self.temperature
                raw_logits = new_logits
            else:
                raw_logits = (logits-priors)/self.temperature
                new_logits = priors + raw_logits
            if self.do_entropy:
                raw_logits_normalized = F.log_softmax(raw_logits, dim=1)
                self.entropy = torch.sum(-raw_logits_normalized*torch.exp(raw_logits_normalized), dim=1)

        eff_logits = new_logits
        x = self.gumbel.sample(logits.shape).to(device=device, dtype=eff_logits.dtype) + eff_logits
        _, out = torch.max(x, -1)
        all_logp = F.log_softmax(eff_logits, dim=1)
        self.logp = torch.cat([this_logp[this_ind:(this_ind+1)] for this_logp, this_ind in zip(all_logp, out)])
        return out

    def effective_logits(self, logits):
        return logits

class SoftmaxRandomSamplePolicySparse(SimplePolicy):
    '''
    Randomly samples from the softmax of the logits
    # https://hips.seas.harvard.edu/blog/2013/04/06/the-gumbel-max-trick-for-discrete-distributions/
    TODO: should probably switch that to something more like
    http://pytorch.org/docs/master/distributions.html
    '''
    def __init__(self, do_entropy=False):
        '''

        '''
        super().__init__()
        self.gumbel = Gumbel(loc=0, scale=1)
        self.do_entropy = do_entropy
        if self.do_entropy:
            self.entropy = None

    def forward(self, all_logits_list, all_action_inds_list):
        '''

        :param logits: list of tensors of len batch_size
        :param action_inds: list of long tensors with indices of the actions corresponding to the logits
        :return:
        '''
        # The number of feasible actions is differnt for every item in the batch, so for loop is the simplest
        logp = []
        out_actions = []
        for logits, action_inds in zip(all_logits_list, all_action_inds_list):
            if len(logits):
                x = self.gumbel.sample(logits.shape).to(device=device, dtype=logits.dtype) + logits
                _, out = torch.max(x, -1)
                this_logp = F.log_softmax(logits)[out]
                out_actions.append(action_inds[out])
                logp.append(this_logp)
            else:
                out_actions.append(torch.tensor(0, device=logits.device, dtype=action_inds.dtype))
                logp.append(torch.tensor(0.0, device=logits.device, dtype=logits.dtype))
        self.logp = torch.stack(logp)
        out = torch.stack(out_actions)
        return out

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

