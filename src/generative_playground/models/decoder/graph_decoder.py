import torch

from generative_playground.codec.hypergraph_grammar import HypergraphGrammar
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
                 mask_gen: DoubleMaskGen,
                 grammar: HypergraphGrammar,
                 node_selection_policy: SimplePolicy):
        super().__init__()
        self.mask_gen = mask_gen
        self.graph_encoder = GraphEmbedder(grammar=grammar)
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
        self.mask_gen.apply_action(last_action)
        node_mask = self.mask_gen.valid_node_mask(max_nodes = self.graph_encoder.output_shape[1])

        # select the next node to expand
        model_out = self.model(self.mask_gen.graphs) # batch x num_nodes, batch x num_nodes x num_actions
        next_node = self.node_selection_policy(model_out['node'] - 1e-6 * (1 - node_mask)) # batch of longs
        self.mask_gen.pick_next_node_to_expand(next_node)

        # and now choose the next logits for the appropriate nodes
        next_logits = model_out['action']
        next_logits_compact = torch.cat([next_logits[b,node,:] for b,node in enumerate(next_node)], dim=0)
        # now that we know which nodes we're going to expand, can generate action masks
        action_mask = self.mask_gen.valid_action_mask()
        masked_logits = next_logits_compact - 1e6 * (1 - action_mask)

        return next_node, masked_logits


class GraphEncoder(nn.Module):
    def __init__(self,
                 grammar,
                 max_nodes=70,
                 d_model=512,
                 drop_rate=0.1):
        super().__init__()
        self.embedder = GraphEmbedder(max_nodes, d_model, grammar)
        # TODO: get the transformer parameters from model_settings, also d_model
        self.transformer = TransformerEncoder(n_src_vocab=None,
                                              n_max_seq=max_nodes,
                                              d_model=d_model,
                                              dropout=drop_rate,
                                              embedder=self.embedder,
                                              n_layers=6,
                                              n_head=8,
                                              d_k=64,
                                              d_v=64)
        self.output_shape = self.transformer.output_shape

    def forward(self, graphs):
        return self.transformer(graphs)



