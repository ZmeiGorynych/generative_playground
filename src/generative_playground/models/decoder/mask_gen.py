import torch
from torch import nn as nn

from generative_playground.utils.gpu_utils import to_gpu


class DummyMaskGenerator(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.num_actions = num_actions

    def forward(self, last_action):
        '''
        Consumes one action at a time, responds with the mask for next action
        : param last_action: ints of shape (batch_size) previous action ; should be [None]*batch_size for the very first step
        '''
        return to_gpu(torch.ones(len(last_action),self.num_actions))

    def reset(self):
        '''
        Reset any internal state, in order to start on a new sequence
        :return:
        '''
        pass


class DoubleMaskGen:
    pass