import torch
from torch import nn as nn

from generative_playground.utils.gpu_utils import device, to_gpu


class Stepper(nn.Module):
    '''
    Abstract parent of steppers
    '''
    def __init__(self, feature_len=None, max_seq_len=float('inf')):
        super().__init__()
        self.output_shape = [None, feature_len]
        self.n = 0
        self.max_seq_length = max_seq_len
        self.z = None

    def init_encoder_output(self, z):
        '''
        initialize
        :param z:
        :return:
        '''
        self.z = z
        self.n = 0
        if z is not None and len(z.size()) >=1: # we got an actual batch of data, as opposed to None
            self.output_shape = [z.size()[0], self.output_shape[1]]

    def register_step(self):
        self.n += 1
        if self.n >= self.max_seq_length:
            raise StopIteration()

    def forward(self, *args, **kwargs):
        raise NotImplementedError()


class DummyStepper(Stepper):
    def __int__(self,
                feature_len=10,
                max_seq_len=None):
        super.__init__(feature_len, max_seq_len)
        self.output_shape[1] = feature_len
        self.max_seq_len = max_seq_len

    def forward(self, *args, **kwargs):
        self.register_step()
        return torch.zeros(*self.output_shape, device = device)


class RandomDecoder(Stepper):
    def __init__(self,
                 feature_len=None,
                 max_seq_length=None):
        super().__init__(feature_len, max_seq_length)
        # self.feature_len = feature_len
        # this just there so the optimizer sees some params to optimize
        self.dummy_fc = nn.Linear(feature_len,feature_len)

    def forward(self, last_action, *args, **kwargs):
        '''
        One step of the RNN model
        :param last_action: batch of ints, all equaling None for first step
        :return: batch x feature_len zeros
        '''
        self.register_step()
        if self.output_shape[0] is None:
            self.output_shape[0] = len(last_action)
        return self.dummy_fc(to_gpu(torch.zeros(*self.output_shape)))

    # def init_encoder_output(self,z):
    #     '''
    #     Must be called at the start of each new sequence
    #     :param z:
    #     :return:
    #     '''
    #     self.n = 0