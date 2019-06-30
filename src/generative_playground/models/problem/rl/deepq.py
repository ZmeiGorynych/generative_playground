from collections import deque
import math
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from generative_playground.codec.hypergraph import HyperGraph
from generative_playground.models.losses.wasserstein_loss import WassersteinLoss
from generative_playground.models.discrete_distribution_utils import *

def slice(x, b):
    out = [xx[b] for xx in x]
    # out[0] = [out[0]] # need to wrap these so Pytorch loader can concatenate them
    assert len(out) == 3
    assert isinstance(out[0], HyperGraph) or out[0] is None, "First element of the state tuple must be None or a HyperGraph"
    assert isinstance(out[1], np.ndarray), "Second element of the state tuple must be an ndarray"
    assert isinstance(out[2], np.ndarray), "Third element of the state tuple must be an ndarray"
    return out



class QLearningDataset(deque):
    def update_data(self, new_data):
        batch_size, num_steps = new_data['rewards'].shape
        for s in range(num_steps):
            for b in range(batch_size):
                old_state = new_data['env_outputs'][s][0]
                if old_state[0][b] is None or len(old_state[0][b].nonterminal_ids()):  # exclude padding states
                    new_state = new_data['env_outputs'][s+1][0]
                    reward = new_data['rewards'][b:b+1, s]
                    action = new_data['actions'][s][b]
                    exp_tuple =(slice(old_state, b),
                                 action,
                                 reward,
                                 slice(new_state,b))
                    self.append(exp_tuple)

def pad_numpy(x, max_len, pad_value):
    if len(x) >= max_len: # sometimes it's been padded too much as part of its batch, need to trim that
        return x[:max_len]
    else:
        return np.concatenate([x, pad_value*np.ones([max_len-len(x),*x.shape[1:]])], axis=0)

def collate_states(states_batch):
    graphs = [b[0] for b in states_batch]
    max_len = max([1 if g is None else len(g) for g in graphs])
    node_priors = np.array([pad_numpy(b[1], max_len=max_len, pad_value=-1e5) for b in states_batch])
    full_priors = np.array([pad_numpy(b[2], max_len=max_len, pad_value=-1e5) for b in states_batch])
    assert max_len == node_priors.shape[1]
    assert max_len == full_priors.shape[1]
    assert len(full_priors.shape) == 3
    return (graphs, np.array(node_priors), np.array(full_priors))

def collate_experiences(batch):
    old_states = collate_states([b[0] for b in batch])
    new_states = collate_states([b[3] for b in batch])
    actions = [b[1] for b in batch]
    rewards = torch.cat([b[2] for b in batch])
    assert old_states[0][0] != new_states[0][0], "Invalid experience tuple: graph unchanged" # test that this is a valid state transition
    return old_states, actions, rewards, new_states


class DeepQModelWrapper(nn.Module):
    def __init__(self, model, gamma=1.0):
        super().__init__()
        # the value prediction model, returns an array of exp value for node x rule choice
        # we interpret its output as logits to feed to a tanh, assuming the actual values are between -1 and 1
        self.model = model
        self.gamma = gamma

    def forward(self, inputs):
        # TODO: review, does this correspond to the new way of doing things?
        old_state, actions, reward, new_state = inputs
        # make the target
        new_values = F.tanh(self.model(new_state[0], new_state[2])['action'].detach())
        new_mask = (torch.from_numpy(new_state[2]) > -1e4).to(dtype=reward[0].dtype,
                                                              device=reward[0].device)
        post_value = (new_values*new_mask).max(dim=2)[0].max(dim=1)[0]
        # TODO: in our case, if terminal use reward, only otherwise use post_value
        target = reward + self.gamma * post_value

        # evaluate the model prediction for the pre-determined action
        model_out = F.tanh(self.model(old_state[0], old_state[2])['action'])
        batch_size = model_out.size(0)
        model_selected = torch.stack([model_out.view(batch_size, -1)[b, a] for b, a in enumerate(actions)])
        return {'out': model_selected, 'target': target}


class DeepQLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, outputs):
        # TODO: here we'd have the Wasserstein penalty for distribution targets
        return self.loss(outputs['out'], outputs['target'])

class DistributionaDeepQModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        # the value prediction model, returns an array of exp value for node x rule choice
        # we interpret its output as logits to feed to a tanh, assuming the actual values are between -1 and 1
        self.model = model
        # self.prob_calc = SoftmaxPolicyProbsFromDistributions() # TODO: plug in thompson probs instead

    def forward(self, inputs):
        old_state, actions, reward, new_state = inputs
        # evaluate the model distribution prediction for the pre-determined action
        model_out = self.model(old_state[0], old_state[2])['action_distrs']
        model_selected = torch.cat([model_out[b:(b+1),a,:] for b, a in enumerate(actions)]) # batch x bins

        # now calculate the targets
        new_out = self.model(new_state[0], new_state[2])
        new_values = new_out['action_distrs'].detach()# batch x action x bins
        masked_policy_logits = new_out['masked_policy_logits'].detach()
        aggr_targets = aggregate_distributions_by_policy_logits(new_values, masked_policy_logits)

        # make the target
        target = torch.zeros_like(model_selected)
        for b in range(len(target)):
            if reward[b] > 0:  # TODO: check for whether it's a terminal state instead
                target[b,:] = to_bins(reward[b], len(target[b,:]), target[b,:])
            else:
                target[b,:] = aggr_targets[b, :]

        return {'out': model_selected, 'target': target}


class DistributionalDeepQWassersteinLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = WassersteinLoss()

    def forward(self, outputs):
        return self.loss(outputs['out'], outputs['target'])