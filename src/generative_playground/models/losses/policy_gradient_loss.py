import torch
from torch import nn as nn
import torch.nn.functional as F


class PolicyGradientLoss(nn.Module):
    def __init__(self, loss_type='mean', loss_cutoff=1e4, keep_records=10): #last_reward_wgt=0.0,
        super().__init__()
        self.loss_type = loss_type
        self.loss_cutoff = loss_cutoff
        #self.last_reward_wgt = last_reward_wgt
        # self.sm_reward = None
        self.best_rewards = set([float('-inf')])
        self.keep_records = keep_records

    def update_record_list(self, new_value):
        worst = min(self.best_rewards)
        if new_value > worst:
            self.best_rewards.add(new_value)
        if len(self.best_rewards) > self.keep_records:
            self.best_rewards.discard(worst)

    def forward(self, model_out):
        '''
        Calculate a policy gradient loss given the logits
        :param logits: batch x seq_length x num_actions floats
        :param actions: batch x seq_length ints
        :param rewards: batch x seq_len floats
        :param terminals: batch x seq_len True if sequence has terminated at that step or earlier
        :return: float loss
        '''
        # actions, logits, rewards, terminals, info = model_out
        smiles, valid = model_out['info']

        if 'logp' in model_out:
            total_logp = model_out['logp']
        else:  # old-style outputs
            _, seq_len, _ = model_out['logits'].size()
            float_type = model_out['logits'].dtype
            log_p = F.log_softmax(model_out['logits'], dim=2) * (1-model_out['terminals'].unsqueeze(2).to(float_type))

            valid = valid.to(dtype=log_p.dtype)
            #total_rewards = total_rewards/total_rewards.mean() # normalize to avg weight 1
            total_logp = 0
            for i in range(seq_len):
                dloss = torch.diag(-log_p[:, i, model_out['actions'][:,i]]) # batch_size, hopefully
                total_logp += dloss
        if len(model_out['rewards'].shape) > 1: # old-style outputs
            total_rewards = model_out['rewards'].sum(1).to(dtype=log_p.dtype)
        else:
            total_rewards = model_out['rewards']

        my_loss = 0
        # loss_cutoff causes us to ignore off-policy examples that are grammatically possible but masked away
        best_ind = torch.argmax(total_rewards)
        best_loss = total_logp[best_ind]
        if 'mean' in self.loss_type:
            rewardloss = (total_logp * total_rewards)[total_logp < self.loss_cutoff]
            mean_loss = rewardloss.mean()/(total_rewards.abs().mean()+1e-6)
            my_loss += mean_loss
        if 'advantage' in self.loss_type:
            # if self.sm_reward is None: # unnecessary complication, the batch average reward is quite stable over time
            #     self.sm_reward = total_rewards.mean()
            # else:
            #     self.sm_reward = self.last_reward_wgt*self.sm_reward + (1-self.last_reward_wgt)*total_rewards.mean()
            adv = total_rewards - total_rewards.mean() # self.sm_reward
            rewardloss = (total_logp * adv)[total_logp < self.loss_cutoff]
            mean_loss = rewardloss.mean() / (adv.abs().mean()+1e-5)
            my_loss += mean_loss
        if 'record' in self.loss_type: # all rewards that exceed the best 10 observed so far, get their own loss contrib
            first_time = len(self.best_rewards) == 1
            min_record = min(self.best_rewards)
            for i, reward in enumerate(total_rewards):
                if reward > min_record:
                    if not first_time:
                        my_loss += total_logp[i]
                    self.update_record_list(reward.data.item())

        if 'best' in self.loss_type:
            if valid[best_ind] == 0:
                best_loss *= 0.0
            my_loss += best_loss
        if 'valid' in self.loss_type:
            valid_reward = 2*valid - 1
            valid_loss = (valid_reward*total_logp).mean()
            my_loss += valid_loss
        # check for NaNs
        assert(my_loss == my_loss)
        if sum(valid) > 0:
            self.metrics = {'rec rwd': max(self.best_rewards),
                            'avg rwd': total_rewards.mean().data.item(),
                            'max rwd': total_rewards.max().data.item(),
                            'med rwd': total_rewards.median().data.item()
                            }
        else:
            self.metrics = {}

        try:
            smiles = model_out['info'][0]
            self.metrics.update({'unique': len(set(smiles)) / len(smiles)})
        except:
            pass
        return my_loss