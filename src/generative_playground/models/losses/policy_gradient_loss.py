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
        actions, logits, rewards, terminals = model_out
        batch_size, seq_len, num_actions = logits.size()
        log_p = F.log_softmax(logits, dim=2) * (1-terminals.unsqueeze(2))
        total_rewards = rewards.sum(1)
        total_loss = 0
        for i in range(seq_len):
            dloss = torch.diag(-log_p[:, i, actions[:,i]]) # batch_size, hopefully
            total_loss += dloss

        total_loss *= total_rewards
        self.metrics = {'avg reward': total_rewards.mean().data.item(),
                        'max reward': total_rewards.max().data.item()}
        if self.loss_type == 'mean':
            my_loss = total_loss.mean()
        elif self.loss_type == 'best':
            best_ind = torch.argmax(total_rewards)
            my_loss = total_loss[best_ind]

        return my_loss