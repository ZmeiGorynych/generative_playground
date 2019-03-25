import torch.nn as nn
from generative_playground.utils.gpu_utils import device
import torch
from generative_playground.models.decoder.stepper import Stepper
from generative_playground.models.decoder.policy import SimplePolicy
from generative_playground.models.decoder.mask_gen import DoubleMaskGen

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

        mask = FloatTensor(self.mask_gen(last_action))
        masked_logits = next_logits - 1e6 * (1 - mask)
        return masked_logits


class DoubleMaskingHead(Stepper):
    def __init__(self, model: Stepper,
                 mask_gen: DoubleMaskGen,
                 node_selection_policy: SimplePolicy):
        super().__init__()
        self.mask_gen = mask_gen
        self.model = model
        self.output_shape = self.model.output_shape
        self.node_selection_policy = node_selection_policy

    def init_encoder_output(self, z):
        '''
        Must be called at the start of each new sequence
        :param z: encoder output
        :return: None
        '''
        self.mask_gen.reset()
        self.model.init_encoder_output(z)

    def forward(self, last_action):
        '''

        :param last_action: batch of longs
        :return:
        '''
        # update the graph with the rule expansion chosen -
        # it remembers the node to apply it to from the previous step
        self.mask_gen.choose_action(last_action).to(device)

        # choose the next node to expand
        next_node_logits, next_logits = self.model(last_action) # batch x num_nodes, batch x num_nodes x num_actions
        node_mask = self.mask_gen.valid_node_mask()
        next_node = self.node_selection_policy(next_node_logits - 1e-6 * (1 - node_mask)) # batch of longs

        # and now choose the next logits for the appropriate nodes
        next_logits_compact = torch.cat([next_logits[b,node,:] for b,node in enumerate(next_node)], dim=0)
        # now that we know which nodes we're going to expand, can generate action masks
        action_mask = self.mask_gen.valid_action_mask()
        masked_logits = next_logits - 1e6 * (1 - action_mask)

        return next_node, masked_logits