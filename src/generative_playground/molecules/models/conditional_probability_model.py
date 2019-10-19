import torch.nn as nn
import torch
import numpy as np
from generative_playground.utils.gpu_utils import device
from generative_playground.codec.hypergraph_mask_generator import mask_from_cond_tuple
from generative_playground.codec.hypergraph_grammar import HypergraphGrammar

class CondtionalProbabilityModel(nn.Module):
    def __init__(self, grammar: HypergraphGrammar):
        super().__init__()
        self.grammar = grammar
        num_rules = len(grammar.rules)
        num_conditionals = len(grammar.conditional_frequencies)
        self.unconditionals = nn.Parameter(torch.zeros(num_rules))
        self.conditionals = nn.Parameter(torch.zeros(num_conditionals, num_rules))
        self.condition_to_ind = {pair: ind for ind, pair in enumerate(grammar.conditional_frequencies.keys())}
        self.ind_to_condition = {ind: pair for pair, ind in self.condition_to_ind.items()}
        # self.ind_to_nonterminal = None # TODO: implement
        self.mask_by_cond_tuple = np.zeros((list(self.conditionals.shape)), dtype=np.int)
        for ind, cond in self.ind_to_condition.items():
            self.mask_by_cond_tuple[ind, :] = mask_from_cond_tuple(self.grammar, cond)
        self.torch_mask_by_cond_tuple = torch.from_numpy(self.mask_by_cond_tuple).to(
            device=self.unconditionals.device)
        self.output_shape = {'action': [None]}
        self.to(device=device)

    def get_params_as_vector(self):
        total_probs = (self.unconditionals + self.conditionals).cpu().detach().numpy()
        out = total_probs[self.mask_by_cond_tuple==1]
        return out

    def set_params_from_vector(self, in_vector):
        self.unconditionals.data[:] = 0
        self.conditionals.data[self.torch_mask_by_cond_tuple==1] = torch.from_numpy(in_vector).to(
                                                                                 dtype=self.unconditionals.dtype,
                                                                                 device=self.unconditionals.device
                                                                                 )

    def forward(self, graphs, full_logit_priors):
        max_nodes = max([1 if g is None else len(g) for g in graphs])
        # action logit padding is -1e5 so we never choose those states
        action_logits = -1e5*torch.ones(len(graphs), max_nodes, len(self.grammar)).to(device=self.conditionals.device,
                                                                                      dtype=self.conditionals.dtype)
        for ig,g in enumerate(graphs):
            action_logits[ig] = self.conditional_logits(g, action_logits[ig])

        masked_logits, used_priors = self.handle_priors(action_logits.view(len(graphs),-1),
                                                        full_logit_priors)
        out = {}
        out['masked_policy_logits'] = masked_logits
        out['used_priors'] = used_priors
        return out

    def conditional_logits(self, g, out):
        if g is None: # first step, no graph yet
            query = (None, None)
            cond_ind = self.condition_to_ind[query]
            assert len(out) == 1, "A None Graph only has one 'node' to choose from!"
            out[0] = self.unconditionals + self.conditionals[cond_ind]
        else:
            child_ids = g.child_ids()
            for n, node_id in enumerate(g.node.keys()):
                if node_id in child_ids:
                    child = g.node[node_id]
                    query = (child.rule_id, child.node_index)
                    cond_ind = self.condition_to_ind[query]
                    out[n] = self.unconditionals + self.conditionals[cond_ind]
        return out

    def handle_priors(self, next_logits, full_logit_priors):
        dtype = next_logits.dtype
        batch_size = len(next_logits)
        full_priors_pytorch = torch.from_numpy(full_logit_priors).to(device=next_logits.device,
                                                                     dtype=dtype).view(batch_size, -1)
        return next_logits + full_priors_pytorch, full_priors_pytorch
