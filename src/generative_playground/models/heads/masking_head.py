import torch
from generative_playground.models.decoder.stepper import Stepper
from generative_playground.utils.gpu_utils import device

class MaskingHead(Stepper):
    def __init__(self, model: Stepper, mask_gen):
        super().__init__()
        self.mask_gen = mask_gen
        self.model = model
        self.output_shape = self.model.output_shape

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

        if 'last_action' in kwargs:
            last_action = kwargs['last_action']
        else:
            last_action = args[0]

        mask = torch.from_numpy(self.mask_gen(last_action)).to(device=device, dtype=next_logits.dtype)
        masked_logits = next_logits - 1e4 * (1 - mask)
        return masked_logits


