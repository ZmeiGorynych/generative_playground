from collections import deque
import torch.nn as nn
import torch

class QLearningDataset(deque):
    def update_data(self, new_data):
        batch_size, num_steps, _ = new_data['rewards'].shape
        for b in range(batch_size):
            for s in range(num_steps):
                # TODO check for padding
                old_state = new_data['env_outputs'][s][0][b]
                new_state = new_data['env_outputs'][s+1][0][b]
                reward = new_data['rewards'][b,s]
                action = [new_data['actions'][s][0][b], new_data['actions'][s][1][b]]
                new_mask = new_data['env_outputs'][s+1][2][b] > -1e4 # ones are the good ones
                self.data.append((old_state, action, reward, new_state, new_mask))


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

