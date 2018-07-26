import torch.nn as nn
from generative_playground.utils.gpu_utils import FloatTensor
import torch

class MaskingHead(nn.Module):
    def __init__(self, model, mask_gen):
        super().__init__()
        self.mask_gen = mask_gen
        self.model = model

    def init_encoder_output(self, z):
        '''
        Must be called at the start of each new sequence
        :param z: encoder output
        :return: None
        '''
        self.mask_gen.reset()
        self.model.init_encoder_output(z)

    def forward(self, *args, **kwargs):
        '''

        :param args:
        :param kwargs:
        :return:
        '''
        next_logits = self.model(*args, **kwargs)
        # just in case we were returned a sequence of length 1 rather than a straight batch_size x num_actions
        next_logits = torch.squeeze(next_logits, 1)

        mask = FloatTensor(self.mask_gen(kwargs['last_action']))
        masked_logits = next_logits - 1e4 * (1 - mask)
        return masked_logits

