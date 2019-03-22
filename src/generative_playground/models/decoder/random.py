import torch.nn as nn
import torch
from generative_playground.utils.gpu_utils import to_gpu

class RandomDecoder(nn.Module):
    def __init__(self,
                 feature_len=None,
                 max_seq_length=None):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.feature_len = feature_len
        self.output_shape = [None, feature_len]
        # this just there so the optimizer sees some params to optimize
        self.dummy_fc = nn.Linear(feature_len,feature_len)

    def forward(self, last_action, *args, **kwargs):
        '''
        One step of the RNN model
        :param last_action: batch of ints, all equaling None for first step
        :return: batch x feature_len zeros
        '''
        # check we don't exceed max sequence length
        self.n += 1
        if self.n == self.max_seq_length:
            raise StopIteration()

        return self.dummy_fc(to_gpu(torch.zeros(len(last_action), self.feature_len)))

    def init_encoder_output(self,z):
        '''
        Must be called at the start of each new sequence
        :param z:
        :return:
        '''
        self.n = 0
