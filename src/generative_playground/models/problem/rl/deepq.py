from collections import deque
import torch.nn as nn
import torch
import numpy as np
from generative_playground.codec.hypergraph import HyperGraph

def slice(x, b):
    out = [xx[b] for xx in x]
    assert len(out) == 3
    assert isinstance(out[0], HyperGraph)
    assert isinstance(out[1], np.ndarray)
    assert isinstance(out[2], np.ndarray)

class QLearningDataset(deque):
    def update_data(self, new_data):
        batch_size, num_steps = new_data['rewards'].shape
        for s in range(num_steps):
            for b in range(batch_size):
                # TODO check for padding
                old_state = new_data['env_outputs'][s][0]
                new_state = new_data['env_outputs'][s+1][0]
                reward = new_data['rewards'][b,s]
                action = new_data['actions'][s][b]
                exp_tuple =(slice(old_state,b),
                             action,
                             reward,
                             slice(new_state,b))
                self.append(exp_tuple)


class DeepQModelWrapper(nn.Module):
    def __init__(self, model, gamma=1.0):
        super().__init__()
        self.model = model # the value prediction model, returns an array of exp value for node x rule choice
        self.gamma = gamma

    def forward(self, inputs):
        old_state, action, reward, new_state, new_mask = inputs
        # make the target
        new_values = self.model(new_state)['value'].detach().view(-1,1)
        post_value = new_values[new_mask.view(-1,1)].max()
        target = reward + self.gamma * post_value

        # evaluate the model prediction for the pre-determined action
        model_out = self.model(old_state)
        model_selected = torch.gather(model_out['value'], action) # TODO make this actually work
        return {'out': model_selected, 'target': target}


class DeepQLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, outputs):
        return nn.MSELoss(outputs['out'], outputs['target'])

