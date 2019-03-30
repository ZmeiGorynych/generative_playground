import numpy as np
import torch

from generative_playground.codec.hypergraph_grammar import HypergraphGrammar, HypergraphMaskGenerator
from generative_playground.models.decoder.mask_gen import DoubleMaskGen
from generative_playground.models.decoder.policy import SimplePolicy
from generative_playground.models.decoder.stepper import Stepper
from generative_playground.utils.gpu_utils import device
from generative_playground.models.embedder.graph_embedder import GraphEmbedder
from generative_playground.models.transformer.Models import TransformerEncoder
from torch import nn


class GraphDecoder(Stepper):
    def __init__(self,
                 model,
                 mask_gen: HypergraphMaskGenerator,
                 # node_selection_policy=PickLastValidValue()
                 ):
        super().__init__()
        self.mask_gen = mask_gen
        self.model = model
        self.output_shape = [None, self.model.output_shape['action'][-1]] # a batch of logits to select next action
        # self.node_selection_policy = node_selection_policy

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
        self.mask_gen.apply_action(last_action)
        model_out = self.model(self.mask_gen.graphs)  # batch x num_nodes, batch x num_nodes x num_actions
        node_mask = self.mask_gen.valid_node_mask(max_nodes=self.model.output_shape['node'][1])

        # this is a hacky workaround to always select the last valid node, will deal with arbitrary node selection later
        next_node = []
        for row in node_mask:
            last_ind = -1
            for i, valid in enumerate(row):
                if valid:
                    last_ind = i
            next_node.append(last_ind)
        next_node = np.array(next_node)
        # # select the next node to expand: this presents a bunch of complications, do it later
        # will need to: include the log(P) of the node choice into loss function
        # include node choice into rule encoding so can encode-decode the sequence incl rule choices
        # next_node = self.node_selection_policy(model_out['node'].squeeze(2) - 1e-6 * (1 - node_mask))  # batch of longs
        self.mask_gen.pick_next_node_to_expand(next_node)

        # and now choose the next logits for the appropriate nodes
        next_logits = model_out['action']
        next_logits_compact = torch.cat([next_logits[b, node, :].unsqueeze(0) for b, node in enumerate(next_node)], dim=0)
        # now that we know which nodes we're going to expand, can generate action masks
        action_mask = self.mask_gen.valid_action_mask()
        action_mask_pytorch = torch.from_numpy(action_mask).to(device=device, dtype=torch.float32)
        masked_logits = next_logits_compact - 1e6 * (1 - action_mask_pytorch)

        return masked_logits # will also want to return which node we picked, once we enable that


class GraphEncoder(nn.Module):
    def __init__(self,
                 grammar,
                 max_nodes=70,  # TODO: remove this parameter altogether
                 d_model=512,
                 drop_rate=0.1,
                 transformer_params={'n_layers': 6,
                                     'n_head': 8,
                                     'd_k': 64,
                                     'd_v': 64}):
        super().__init__()
        self.embedder = GraphEmbedder(max_nodes, d_model, grammar)
        # TODO: get the transformer parameters from model_settings, also d_model
        self.transformer = TransformerEncoder(n_src_vocab=None,
                                              n_max_seq=max_nodes,
                                              d_model=d_model,
                                              dropout=drop_rate,
                                              embedder=self.embedder,
                                              **transformer_params)
        self.output_shape = self.transformer.output_shape

    def forward(self, graphs):
        '''
        Embeds graphs and feeds them through the Transformer encoder
        :param graphs: a list of HyperGraphs of length batch
        :return: batch x max_nodes x d_model float32s
        '''
        return self.transformer(graphs)
