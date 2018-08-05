import torch
from torch import nn as nn
import torch.nn.functional as F


class PolicyGradientLoss(nn.Module):
    def __init__(self, loss_type='mean'):
        super().__init__()
        self.loss_type = loss_type

    def forward(self, model_out):
        '''
        Calculate a policy gradient loss given the logits
        :param logits: batch x seq_length x num_actions floats
        :param actions: batch x seq_length ints
        :param rewards: batch x seq_len floats
        :param terminals: batch x seq_len True if sequence has terminated at that step or earlier
        :return: float loss
        '''
        actions, logits, rewards, terminals, info = model_out
        smiles, valid = info
        batch_size, seq_len, num_actions = logits.size()
        log_p = F.log_softmax(logits, dim=2) * (1-terminals.unsqueeze(2))
        total_rewards = rewards.sum(1)
        total_logp = 0
        for i in range(seq_len):
            dloss = torch.diag(-log_p[:, i, actions[:,i]]) # batch_size, hopefully
            total_logp += dloss

        rewardloss =total_logp * total_rewards
        if sum(valid) > 0:
            self.metrics = {'avg reward': total_rewards.mean().data.item() - 2.5,
                        'max reward': total_rewards.max().data.item() - 2.5}
        else:
            self.metrics = {}
        my_loss = 0

        if 'mean' in self.loss_type:
            mean_loss = rewardloss.mean()
            my_loss += mean_loss
        if 'best' in self.loss_type:
            best_ind = torch.argmax(total_rewards)
            best_loss = rewardloss[best_ind]
            if valid[best_ind] == 0:
                best_loss *= 0.0
            my_loss += best_loss
        if 'valid' in self.loss_type:
            valid_reward = 2*valid - 1
            valid_loss = (valid_reward*total_logp).mean()
            my_loss += valid_loss

        return my_loss