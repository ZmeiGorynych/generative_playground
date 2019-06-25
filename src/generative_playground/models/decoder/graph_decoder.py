import numpy as np
import torch
from math import floor

from generative_playground.codec.hypergraph_grammar import HypergraphMaskGenerator
from generative_playground.models.problem.policy import SoftmaxRandomSamplePolicy
from generative_playground.models.decoder.stepper import Stepper
from generative_playground.utils.gpu_utils import device
from generative_playground.models.embedder.graph_embedder import GraphEmbedder
from generative_playground.models.transformer.Models import TransformerEncoder
from generative_playground.models.encoder.basic_rnn import SimpleRNN
from torch import nn
import torch.nn.functional as F


class GraphDecoderWithNodeSelection(Stepper):
    def __init__(self,
                 model,
                 # node_policy=SoftmaxRandomSamplePolicy(),
                 rule_policy=SoftmaxRandomSamplePolicy(),
                 detach_model_output=False
                 ):
        super().__init__()
        # self.node_policy = node_policy
        self.rule_policy = rule_policy
        self.model = model
        self.detach_model_output = detach_model_output
        self.output_shape = [None, self.model.output_shape['action'][-1]]  # a batch of logits to select next action


    def init_encoder_output(self, z):
        '''
        Must be called at the start of each new sequence
        :param z: encoder output
        :return: None
        '''
        # self.mask_gen.reset()
        self.model.init_encoder_output(z)

    def forward(self, last_state):
        """

        :param last_state: self.mask_gen.graphs
        :param node_mask:
        :param full_logit_priors:
        :return:
        """
        graphs, node_mask, full_logit_priors = last_state
        model_out = self.model(graphs, full_logit_priors)
        masked_logits = model_out['masked_policy_logits']
        used_priors = model_out['used_priors']
        if self.detach_model_output:
            masked_logits = masked_logits.detach()
            used_priors = used_priors.detach()
        # apply policy - would need to make more elaborate if we want to have separate temperatures on node and rule selection
        next_action_ = self.rule_policy(masked_logits, used_priors)
        action_logp = self.rule_policy.logp

        return next_action_, action_logp #(next_node, next_action), action_logp # will also want to return which node we picked, once we enable that

    # def old_forward(self, last_state):
    #     """
    #
    #     :param last_state: self.mask_gen.graphs
    #     :param node_mask:
    #     :param full_logit_priors:
    #     :return:
    #     """
    #     graphs, node_mask, full_logit_priors = last_state
    #
    #     model_out = self.model(graphs)  # batch x num_nodes, batch x num_nodes x num_actions
    #
    #     node_logits = model_out['node'].squeeze(2)
    #     dtype = node_logits.dtype
    #     node_logits += torch.from_numpy(node_mask).to(device=device, dtype=dtype) # batch x max_num_nodes
    #     next_node = self.node_policy(node_logits)
    #     node_selection_logp = torch.cat([F.log_softmax(node_logits, dim=1)[b, node:(node+1)] for b, node in enumerate(next_node)])
    #
    #     # and now choose the next logits for the appropriate nodes
    #     next_logits = model_out['action']
    #     next_logits_compact = torch.cat([next_logits[b, node, :].unsqueeze(0) for b, node in enumerate(next_node)],
    #                                     dim=0)
    #     # now that we know which nodes we're going to expand, can generate action masks: the logit priors also include masking
    #     full_logit_priors_pytorch = torch.from_numpy(full_logit_priors).to(device=device, dtype=dtype)
    #     logit_priors_pytorch = torch.cat([full_logit_priors_pytorch[b, node, :].unsqueeze(0)
    #                                       for b, node in enumerate(next_node)],
    #                                     dim=0)
    #
    #     masked_logits = next_logits_compact + logit_priors_pytorch
    #     next_action = self.rule_policy(masked_logits, logit_priors_pytorch)
    #     action_logp = torch.cat([F.log_softmax(masked_logits, dim=1)[a, action:(action+1)] for a, action in enumerate(next_action)], dim=0)
    #
    #     # we only care about the logits for the logP, right?
    #
    #     return (next_node, next_action), action_logp + node_selection_logp # will also want to return which node we picked, once we enable that

class GraphDecoder(Stepper):
    def __init__(self,
                 model,
                 mask_gen: HypergraphMaskGenerator,
                 # node_selection_policy=PickLastValidValue()
                 ):
        super().__init__()
        self.mask_gen = mask_gen
        self.model = model
        self.output_shape = [None, self.model.output_shape['action'][-1]]  # a batch of logits to select next action
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
        node_mask = self.mask_gen.valid_node_mask()

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
        next_logits_compact = torch.cat([next_logits[b, node, :].unsqueeze(0) for b, node in enumerate(next_node)],
                                        dim=0)
        # now that we know which nodes we're going to expand, can generate action masks
        logit_priors = self.mask_gen.action_prior_logits()  # that includes any priors
        logit_priors_pytorch = torch.from_numpy(logit_priors).to(device=device, dtype=next_logits_compact.dtype)
        masked_logits = next_logits_compact + logit_priors_pytorch

        return masked_logits  # will also want to return which node we picked, once we enable that


class GraphEncoder(nn.Module):
    def __init__(self,
                 grammar,
                 d_model=512,
                 drop_rate=0.1,
                 transformer_params={'n_layers': 5, #6
                                     'n_head': 6, #8
                                     'd_k': 32, # a64
                                     'd_v': 32},
                 rnn_params={'num_layers': 3,
                             'bidirectional': False},
                 model_type='transformer'):
        super().__init__()
        self.embedder = GraphEmbedder(target_dim=d_model, grammar=grammar)
        if model_type == 'transformer':
            # TODO: get the transformer parameters from model_settings, also d_model
            self.encoder = TransformerEncoder(n_src_vocab=None,
                                              d_model=d_model,
                                              dropout=drop_rate,
                                              embedder=self.embedder,
                                              **transformer_params)
        elif model_type == 'rnn':
            self.rnn = SimpleRNN(hidden_n=d_model,
                                     feature_len=d_model,
                                     drop_rate=drop_rate,
                                     **rnn_params)
            self.encoder = nn.Sequential(self.embedder, self.rnn)
            self.encoder.output_shape = self.rnn.output_shape
        else:
            raise ValueError("Unknown model_type " + model_type + ", must be transformer or rnn")
        self.output_shape = self.encoder.output_shape
        self.to(device=device)

    def forward(self, graphs):
        '''
        Embeds graphs and feeds them through the Transformer encoder
        :param graphs: a list of HyperGraphs of length batch
        :return: batch x max_nodes x d_model float32s
        '''
        pre_out = self.encoder(graphs)
        return pre_out
