import torch
from torch import nn as nn

from generative_playground.gpu_utils import to_gpu
from generative_playground.models.decoder.decoders import OneStepDecoderContinuous, SimpleDiscreteDecoder
from generative_playground.models.decoder.policy import SoftmaxRandomSamplePolicy, PolicyFromTarget


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


class DummyNNModel(nn.Module):
    def __init__(self, max_len, num_actions):
        super().__init__()
        self.max_len = max_len
        self.num_actions = num_actions

    def forward(self, z):
        return to_gpu(torch.randn(len(z), self.max_len, self.num_actions))

if __name__ == '__main__':
    batch_size = 25
    max_len = 10
    num_actions = 15
    latent_size = 20

    policy = SoftmaxRandomSamplePolicy()
    mask_gen = DummyMaskGenerator(num_actions)
    stepper = OneStepDecoderContinuous(DummyNNModel(max_len, num_actions))
    decoder = SimpleDiscreteDecoder(stepper, policy, mask_gen)
    z = torch.randn(batch_size, latent_size)
    out_actions, out_logits = decoder(z)
    print('success!')