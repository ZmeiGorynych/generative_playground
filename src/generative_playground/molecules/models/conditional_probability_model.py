import torch.nn as nn
import torch
import numpy as np
from generative_playground.utils.gpu_utils import device
from generative_playground.codec.hypergraph_mask_generator import mask_from_cond_tuple
from generative_playground.codec.hypergraph_grammar import HypergraphGrammar
from collections import OrderedDict


class CondtionalProbabilityModel(nn.Module):
    def __init__(self, grammar: HypergraphGrammar, drop_rate=0.0, sparse_output=True):
        super().__init__()
        self.grammar = grammar
        self.sparse_output = sparse_output
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
        self.dropout = nn.Dropout(1 - drop_rate)
        # self.dropout_mask = torch.ones_like(self.conditionals.data)
        # self.dropout_mask.requires_grad = False
        self.to(device=device)

    def get_params_as_vector(self):
        total_probs = (self.unconditionals + self.conditionals).cpu().detach().numpy()
        out = total_probs[self.mask_by_cond_tuple == 1]
        return out

    def set_params_from_vector(self, in_vector):
        self.unconditionals.data[:] = 0
        self.conditionals.data[self.torch_mask_by_cond_tuple == 1] = torch.from_numpy(in_vector).to(
            dtype=self.unconditionals.dtype,
            device=self.unconditionals.device
        )

    def collapse_unconditionals(self):
        # add the unconditional vectors to the conditional ones, and zero the unconditionals:
        # doesn't change model's output but makes it easier to work with the coefficients
        self.conditionals.data += self.unconditionals.data
        self.unconditionals.data[:] = 0

    def forward(self, graphs, full_logit_priors):
        max_nodes = max([1 if g is None else len(g) for g in graphs])
        # action logit padding is -1e5 so we never choose those states
        action_logits = -1e5 * torch.ones(len(graphs), max_nodes, len(self.grammar)).to(device=self.conditionals.device,
                                                                                        dtype=self.conditionals.dtype)
        for ig, g in enumerate(graphs):
            action_logits[ig] = self.conditional_logits(g, action_logits[ig])

        masked_logits, used_priors = self.handle_priors(action_logits.view(len(graphs), -1),
                                                        full_logit_priors)
        out = {}
        out['masked_policy_logits'] = masked_logits
        out['used_priors'] = used_priors
        if self.sparse_output:
            out['valid_policy_logits'] = []
            out['action_inds'] = []
            ind_array = torch.tensor(list(range(masked_logits.shape[1])),
                                     device=masked_logits.device,
                                     requires_grad=False)
            # todo: the below is to test the flow for sparse evaluation, before moving to sparsely parametrized model
            for i, logits in enumerate(masked_logits):
                this_mask = logits > -1e5
                out['valid_policy_logits'].append(logits[this_mask])
                out['action_inds'].append(ind_array[this_mask])

        return out

    def conditional_logits(self, g, out):
        # TODO: implement dropout
        if g is None:  # first step, no graph yet
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


class CondtionalProbabilityModelSparse(nn.Module):
    def __init__(self, grammar: HypergraphGrammar, drop_rate=0.0):
        super().__init__()
        self.grammar = grammar
        self.drop_rate = drop_rate
        self.num_rules = len(grammar.rules)
        self.unconditionals = nn.Parameter(torch.zeros(self.num_rules))

        self.conditionals = nn.ParameterDict()
        self.priors = OrderedDict()
        self.indices = OrderedDict()

        for cond in grammar.conditional_frequencies.keys():
            mask = mask_from_cond_tuple(self.grammar, cond)
            mask_inds = np.nonzero(mask)[0]
            masked_freqs = self.grammar.get_conditional_log_frequencies_single_query(cond)[mask_inds]
            assert len(mask_inds) > 0
            assert len(mask_inds) == len(masked_freqs)
            self.priors[cond] = torch.from_numpy(masked_freqs).to(dtype=self.unconditionals.dtype,
                                                                  device=device)
            assert self.priors[cond].requires_grad is False
            # ParameterDict only accepts strings as j
            self.conditionals[str(cond)] = nn.Parameter(torch.from_numpy(np.zeros_like(masked_freqs)).to(
                dtype=self.unconditionals.dtype))
            assert self.conditionals[str(cond)].requires_grad is True
            self.indices[cond] = torch.from_numpy(mask_inds).to(device=device)

        self.output_shape = {'action': [None]}

        self.to(device=device)

    def joint_logits(self, cond):
        out = self.conditionals[str(cond)] + self.unconditionals[self.indices[cond]]
        return out

    def get_params_as_vector(self):
        out = torch.cat([self.joint_logits(cond) for cond in self.priors.keys()])
        return out.detach().cpu().numpy()

    def set_params_from_vector(self, in_vector):
        self.unconditionals.data[:] = 0
        start = 0
        for cond in self.priors.keys():
            end = start + len(self.indices[cond])
            self.conditionals[str(cond)].data = torch.from_numpy(in_vector[start:end]).to(
                dtype=self.unconditionals.dtype,
                device=self.unconditionals.device)
            start = end

    def collapse_unconditionals(self):
        # add the unconditional vectors to the conditional ones, and zero the unconditionals:
        # doesn't change model's output but makes it easier to work with the coefficients
        for cond in self.conditionals.keys():
            self.conditionals[str(cond)].data += self.unconditionals[self.indices[cond]].data
        self.unconditionals.data[:] = 0

    def forward(self, graphs, full_logit_priors):
        action_logits = [None] * len(graphs)
        action_inds = [None] * len(graphs)

        for ig, g in enumerate(graphs):
            action_logits[ig], action_inds[ig] = self.conditional_logits(g, full_logit_priors[ig])

        out = {'valid_policy_logits': action_logits,
               'action_inds': action_inds}

        return out

    def conditional_logits(self, g, logit_priors):
        # TODO: implement dropout
        if g is None:  # first step, no graph yet
            query = (None, None)
            action_logits = self.joint_logits(query) + self.priors[query]
            action_inds = self.indices[query]
        else:
            all_logits = []
            all_inds = []
            child_ids = set(g.child_ids())
            if len(child_ids):
                for n, node_id in enumerate(g.node.keys()):
                    if node_id in child_ids:
                        child = g.node[node_id]
                        query = (child.rule_id, child.node_index)
                        these_logits = self.joint_logits(query) + self.priors[query]
                        all_logits.append(these_logits)
                        all_inds.append(self.indices[query] + n * self.num_rules)
                action_logits = torch.cat(all_logits)
                action_inds = torch.cat(all_inds)
            else:
                action_logits = torch.tensor([],
                                             dtype=self.unconditionals.dtype,
                                             device=self.unconditionals.device)
                action_inds = torch.tensor([],
                                           dtype=self.indices[(None, None)].dtype,
                                           device=self.indices[(None, None)].device)
        # now apply the mask inferred from the priors - this just gives you the terminal distance filter
        torch_priors = torch.from_numpy(logit_priors).to(device=action_logits.device,
                                                         dtype=action_logits.dtype).view(-1)
        prior_mask = torch_priors[action_inds] > -5e4
        action_logits = action_logits[prior_mask]
        action_inds = action_inds[prior_mask]

        return action_logits, action_inds


class ConditionalModelBlended(nn.Module):
    def __init__(self, grammar: HypergraphGrammar, drop_rate=0.0):
        super().__init__()
        self.old_model = CondtionalProbabilityModel(grammar, drop_rate, sparse_output=True)
        self.sparse_model = CondtionalProbabilityModelSparse(grammar, drop_rate)
        self.output_shape = {'action': [None]}
        self.init_encoder_output = lambda x: None

    def forward(self, graphs, full_logit_priors):
        out1 = self.old_model(graphs, full_logit_priors)
        out2 = self.sparse_model(graphs, full_logit_priors)
        for ig in range(len(graphs)):
            assert ((len(out1['action_inds'][ig]) == 0 and len(out2['action_inds'][ig]) == 0) or
                all(out1['action_inds'][ig] == out2['action_inds'][0])
                )
        return out1
