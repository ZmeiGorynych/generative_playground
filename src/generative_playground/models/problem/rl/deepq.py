from collections import deque
import torch.nn as nn
import torch
import numpy as np
from generative_playground.codec.hypergraph import HyperGraph

def slice(x, b):
    out = [xx[b] for xx in x]
    # out[0] = [out[0]] # need to wrap these so Pytorch loader can concatenate them
    assert len(out) == 3
    assert isinstance(out[0], HyperGraph)
    assert isinstance(out[1], np.ndarray)
    assert isinstance(out[2], np.ndarray)
    return out

class QLearningDataset(deque):
    def update_data(self, new_data):
        batch_size, num_steps = new_data['rewards'].shape
        for s in range(num_steps):
            for b in range(batch_size):
                # TODO check for padding
                old_state = new_data['env_outputs'][s][0]
                new_state = new_data['env_outputs'][s+1][0]
                reward = new_data['rewards'][b:b+1,s]
                action = new_data['actions'][s][b]
                exp_tuple =(slice(old_state,b),
                             action,
                             reward,
                             slice(new_state,b))
                self.append(exp_tuple)

def pad_numpy(x, max_len, pad_value):
    assert len(x) <= max_len, "Input too long for padding!"
    if len(x) == max_len:
        return x
    else:
        return np.concatenate([x, pad_value*np.ones([max_len-len(x),*x.shape[1:]])], axis=0)

def collate_states(states_batch):
    graphs = [b[0] for b in states_batch]
    max_len = max([len(g) for g in graphs])
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
    return old_states, actions, rewards, new_states


class DeepQModelWrapper(nn.Module):
    def __init__(self, model, gamma=1.0):
        super().__init__()
        self.model = model # the value prediction model, returns an array of exp value for node x rule choice
        self.gamma = gamma

    def forward(self, inputs):
        old_state, actions, reward, new_state = inputs
        # make the target
        new_values = self.model(new_state[0])['action'].detach()
        new_mask = (torch.from_numpy(new_state[2]) > -1e4).to(dtype=reward[0].dtype,
                                                              device=reward[0].device)
        post_value = (new_values*new_mask).max(dim=2)[0].max(dim=1)[0]
        target = reward + self.gamma * post_value

        # evaluate the model prediction for the pre-determined action
        model_out = self.model(old_state[0])
        model_selected = torch.stack([model_out['action'][b,a1,a2] for b, (a1,a2) in enumerate(actions)])
        return {'out': model_selected, 'target': target}


class DeepQLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, outputs):
        return self.loss(outputs['out'], outputs['target'])

