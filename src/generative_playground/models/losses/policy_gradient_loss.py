import torch
from torch import nn as nn
import torch.nn.functional as F


class PolicyGradientLoss(nn.Module):
    def __init__(self, loss_type='mean', loss_cutoff=1e5):
        super().__init__()
        self.loss_type = loss_type
        self.loss_cutoff = loss_cutoff

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
        #total_rewards = total_rewards/total_rewards.mean() # normalize to avg weight 1
        total_logp = 0
        for i in range(seq_len):
            dloss = torch.diag(-log_p[:, i, actions[:,i]]) # batch_size, hopefully
            total_logp += dloss

        #total_logp[total_logp > self.loss_cutoff] = 0

        if sum(valid) > 0:
            self.metrics = {'avg reward': total_rewards.mean().data.item(),
                        'max reward': total_rewards.max().data.item()}
        else:
            self.metrics = {}
        my_loss = 0
        # loss_cutoff causes us to ignore off-policy examples that are grammatically possible but masked away
        rewardloss = (total_logp * total_rewards)[total_logp < self.loss_cutoff]
        if 'mean' in self.loss_type:
            mean_loss = rewardloss.mean()/(total_rewards.abs().mean()+1e-8)
            my_loss += mean_loss
        if 'best' in self.loss_type:
            best_ind = torch.argmax(total_rewards)
            best_loss = total_logp[best_ind]# # normalize to weight 1 rewardloss[best_ind]
            if valid[best_ind] == 0:
                best_loss *= 0.0
            my_loss += best_loss
        if 'valid' in self.loss_type:
            valid_reward = 2*valid - 1
            valid_loss = (valid_reward*total_logp).mean()
            my_loss += valid_loss
        # check for NaNs
        assert(my_loss == my_loss)

        return my_loss