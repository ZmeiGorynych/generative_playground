import math
import torch
from torch import nn as nn
import torch.nn.functional as F

def truncated_entropy(x):
    """
    Calculates the entropy capped to
    :param x: matrix of log-probabilities
    :return: capped entropy
    """
    entropy = torch.zeros_like(x)
    big_x = x[x>-1.0]
    entropy[x>-1.0] = torch.exp(big_x)*big_x + math.exp(-1.0)
    return entropy

def entropy_fun(x):
    return x*torch.exp(x)

class PolicyGradientLoss(nn.Module):
    def __init__(self, loss_type='mean', loss_cutoff=1e4, keep_records=10, entropy_wgt=1.0): #last_reward_wgt=0.0,
        super().__init__()
        self.loss_type = loss_type
        self.loss_cutoff = loss_cutoff
        #self.last_reward_wgt = last_reward_wgt
        # self.sm_reward = None
        self.best_rewards = [float('-inf')]
        self.keep_records = keep_records
        self.entropy_wgt = entropy_wgt

    def update_record_list(self, new_value):
        worst = min(self.best_rewards)
        if new_value > worst:
            self.best_rewards.append(new_value)
            self.best_rewards = sorted(self.best_rewards, reverse=True)
        if len(self.best_rewards) > self.keep_records:
            self.best_rewards = self.best_rewards[:self.keep_records]

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
        actions = model_out['actions']



        if 'logp' in model_out:
            # in the old pg code, logp length was for some reason different to actions length
            logp = model_out['logp'][:, :actions.shape[1]]
            entropy = entropy_fun(logp) #truncated_entropy(logp)#[:, :-1]
            entropy[actions == 0] = 0.0 # ignore padding

            if len(logp.shape) > 1:
                total_logp = -logp.sum(1)
                total_entropy = entropy.sum(1)
            else:
                total_logp = -logp
                total_entropy = entropy
        else:  # old-style outputs
            _, seq_len, _ = model_out['logits'].size()
            float_type = model_out['logits'].dtype
            log_p = F.log_softmax(model_out['logits'], dim=2) * (1-model_out['terminals'].unsqueeze(2).to(float_type))

            valid = valid.to(dtype=log_p.dtype)
            #total_rewards = total_rewards/total_rewards.mean() # normalize to avg weight 1
            total_logp = 0
            total_entropy = 0
            for i in range(seq_len):
                this_logp = torch.diag(log_p[:, i, model_out['actions'][:,i]]) # batch_size, hopefully
                total_logp -= this_logp
                total_entropy += torch.exp(this_logp)*this_logp


        if 'entropy' in model_out:
            total_entropy = model_out['entropy'].sum(dim=1)

        if len(model_out['rewards'].shape) > 1: # old-style outputs
            total_rewards = model_out['rewards'].sum(1).to(dtype=total_logp.dtype)
        else:
            total_rewards = model_out['rewards'].to(dtype=total_logp.dtype)
        best_ind = torch.argmax(total_rewards)


        my_loss = self.entropy_wgt * total_entropy.mean()
        if 'mean' in self.loss_type:
            # loss_cutoff causes us to ignore off-policy examples that are grammatically possible but masked away
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

        first_time = len(self.best_rewards) == 1
        min_record = min(self.best_rewards)
        for i, reward in enumerate(total_rewards):
            if reward >= min_record: #note: >= or > makes a big difference?
                # all rewards that exceed the best 10 observed so far, get their own loss contrib
                if 'record' in self.loss_type:
                    if not first_time:
                        my_loss += total_logp[i]
                self.update_record_list(reward.data.item())


        if 'best' in self.loss_type:
            best_loss = total_logp[best_ind]
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
            self.metrics = {'entropy': {'avg_entropy': total_entropy.mean().data.item(),
                                       'best_entropy': total_entropy[best_ind].data.item()}}

            seq_lens = (actions > 0).sum(dim=1)  # do some sequence length stats
            self.metrics['num_steps'] = {'max': seq_lens.max().data.item(),
                                         'median': seq_lens.median().data.item(),
                                         'min': seq_lens.min().data.item(),
                                         'best': seq_lens[best_ind].data.item()}
            if len(logp.shape) > 1:
                best_actions = actions[best_ind, :]
                best_logp = logp[best_ind][best_actions != 0] # ignore padding

                self.metrics['logp'] = {'med_all': logp[actions != 0].median().data.item(),
                                        'max': best_logp.max().data.item(),
                                        'min': best_logp.min().data.item(),
                                        'median':best_logp.median().data.item()}
        else:
            self.metrics = {}



        if my_loss != my_loss:  # NaN check
            print("NaN loss!")
        return my_loss

