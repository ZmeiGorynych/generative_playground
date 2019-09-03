from collections import defaultdict, deque
import math
import numpy as np
from torch.nn import functional as F


class CountRewardAdjuster:
    def __init__(self, reward_fun):
        self.reward_fun = reward_fun
        self.count = defaultdict(int)

    def __call__(self, x):
        raw_reward = self.reward_fun(x)
        self.count[x] += 1
        if raw_reward < 0.0:
            reward = raw_reward*self.count[x]
        elif raw_reward <= 1.0:
            reward = raw_reward**self.count[x]
        else:
            raise ValueError("Reward is supposed to be <=1, got " + str(raw_reward))
        return reward


def originality_mult(zinc_set, history_data, smiles_list):
    out = []
    for s in smiles_list:
        if s in zinc_set:
            out.append(0.5)
        elif s in history_data[0]:
            out.append(0.5)
        elif s in history_data[1]:
            out.append(0.70)
        elif s in history_data[2]:
            out.append(0.85)
        else:
            out.append(1.0)
    return np.array(out)


def sigmoid(x):
    return 1/(1+np.exp(-x))


def discriminator_reward_mult(discrim_model, smiles_list):
    orig_state = discrim_model.training
    discrim_model.eval()
    discrim_out_logits = discrim_model(smiles_list)['p_zinc']
    discrim_probs = F.softmax(discrim_out_logits, dim=1)
    prob_zinc = discrim_probs[:,1].detach().cpu().numpy()
    if orig_state:
        discrim_model.train()
    return prob_zinc


def apply_originality_penalty(extra_repetition_penalty, x, orig_mult):
    assert x <= 1, "Reward must be no greater than 0"
    if x > 0.5: # want to punish nearly-perfect scores less and less
        out = math.pow(x, 1/orig_mult)
    else: # continuous join at 0.5
        penalty = math.pow(0.5, 1/orig_mult) - 0.5
        out = x + penalty

    out -= extra_repetition_penalty*(1-1/orig_mult)
    return out


def adj_reward(discrim_wt, discrim_model, reward_fun_on, zinc_set, history_data, extra_repetition_penalty, x, alt_calc=None):
    if discrim_wt > 1e-5:
        p = discriminator_reward_mult(discrim_model, x)
    else:
        p = 0
    rwd = np.array(reward_fun_on(x))
    orig_mult = originality_mult(zinc_set, history_data, x)
    # we assume the reward is <=1, first term will dominate for reward <0, second for 0 < reward < 1
    # reward = np.minimum(rwd/orig_mult, np.power(np.abs(rwd),1/orig_mult))
    reward = np.array([apply_originality_penalty(extra_repetition_penalty, x, om) for x, om in zip(rwd, orig_mult)])
    out = reward + discrim_wt*p*orig_mult
    # if alt_calc is not None:
    #     alt_out = alt_calc(x)
    #     assert alt_out[0] == out[0]
    return out


def adj_reward_old(discrim_model, p_thresh, randomize_reward, reward_fun_on, zinc_set, history_data, x):
    p = discriminator_reward_mult(discrim_model, x)
    w = sigmoid(-(p-p_thresh)/0.01)
    if randomize_reward:
        rand = np.random.uniform(size=p.shape)
        w *= rand
    reward = np.maximum(reward_fun_on(x), p_thresh)
    weighted_reward = w * p + (1 - w) * reward
    out = weighted_reward * originality_mult(zinc_set, history_data, x)  #
    return out


class AdjustedRewardCalculator:
    def __init__(self, reward_fun, zinc_set, lookbacks, extra_repetition_penalty=0.0, discrim_wt=0, discrim_model=None):
        self.reward_fun = reward_fun
        self.history_data = [deque(['O'], maxlen=lb) for lb in lookbacks]
        self.zinc_set = zinc_set
        self.repetition_penalty = extra_repetition_penalty
        self.discrim_wt = discrim_wt
        self.discrim_model = discrim_model

    def __call__(self, smiles):
        """
        Converts from SMILES inputs to adjusted rewards, keeping track of the molecules observed so far
        :param x: list of SMILES strings
        :return:
        """

        reward = adj_reward(self.discrim_wt, self.discrim_model, self.reward_fun, self.zinc_set, self.history_data, self.repetition_penalty, smiles)
        for s in smiles:  # only store unique instances of molecules so discriminator can't guess on frequency
            for data in self.history_data:
                if s not in data:
                    data.append(s)
        return reward

