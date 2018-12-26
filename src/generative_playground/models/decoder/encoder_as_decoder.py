from torch import nn as nn
from torch.nn import functional as F

from generative_playground.utils.gpu_utils import to_gpu


class EncoderAsDecoder(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.num_steps = encoder.output_shape[1]
        self.output_shape = encoder.output_shape

    def forward(self, z):
        '''

        :param z: batch x num_dims
        :return: multiplex z to num_steps, apply encoder
        '''
        input = z.unsqueeze(1).expand(-1, self.num_steps, -1)
        out = self.encoder(input)
        return None, out # the decoder format assumes first output is a sequence of discrete actions