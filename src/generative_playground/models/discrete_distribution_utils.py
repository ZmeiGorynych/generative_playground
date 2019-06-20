import torch
import torch.nn as nn
import torch.nn.functional as F
import math
eps = 1e-5

def thompson_probabilities(ps, mask):
    """
    Calculates the probability that if we sample from the distributions specified, the value of the a'th variable will be the largess
    :param ps: batch x action x bin non-negative floats
    :param mask: batch x action booleans or ints representing booleans
    :return: batch x action non-negative floats summing up to 1
    """
    batch_size, actions, bins = ps.shape
    out = torch.zeros(batch_size, actions, dtype=ps.dtype, device=ps.device)
    assert all(ps.view(-1)>=0), "Probabilities must be non-negative"
    for b in batch_size:
        for a in actions:
            if mask(b,a):
                assert (ps[a,b].sum() - 1).abs() < eps, "Probabilities for allowed actions must sum up to 1"

    cdfs = ps.cumsum(dim=2)
    for b in batch_size:
        for a in actions:
            if not mask(b,a):
                cdfs[b,a,:] = 1

    # derive the integral formula here
    #TODO: finish this

class SoftmaxPolicyProbsFromDistributions(nn.Module):
    def __init__(self):
        super().__init__()
        self.expected_value_calc = CalcExpectedValue()

    def forward(self, ps, T):
        """
        Takes a set of distributions over the [0,1] interval and translates them into softmax of their expected values
        :param ps: batch x <anything> x bins floats
        :param T: temperature for the softmax
        :return: batch x <anything> floats: probabilities
        """
        exp_values = self.expected_value_calc(ps)
        probs = F.softmax(exp_values/T, dim=-1)
        return probs

class CalcExpectedValue(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = None

    def forward(self, ps):
        """
        Calculates expected values of distributions
        :param ps: batch x <anything> x bins floats: distributions on the [0,1] interval, last element is probability of exactly 1
        :return: batch x <anything> floats: expected values
        """
        if self.w is None or len(self.w) != ps.size(-1):
            self.populate_w(ps)

        out = (ps*self.w).sum(-1)
        return out

    def populate_w(self, ps):
        n = ps.size(-1)
        self.w = torch.zeros(n, device=ps.device, dtype=ps.dtype)
        for i in range(n-1):
            self.w[i] = (i+0.5)/(n-1)
        self.w[-1] = 1


def aggregate_distributions_by_policy(ps, policy_p):
    """
    Calculates the aggregation of probabilities assuming Thompson sampling.
    actions can be one index or actions1, actions2
    :param ps: batch x actions x bins floats distributions per action
    :param policy_p: batch x actions probabilities of each action
    :return: batch x bins floats
    """
    assert all([p >= 0 for p in policy_p.view(-1)])
    policy_p = policy_p/policy_p.sum(-1, keepdim=True)
    assert all([(p - 1.0).abs() < eps for p in policy_p.sum(-1)])
    policy_p = policy_p.unsqueeze(len(policy_p.size()))
    weighted_ps = (ps*policy_p).sum(-2)
    if len(weighted_ps.size()) == 3:
        weighted_ps = weighted_ps.sum(-2)

    assert all([x < eps for x in (weighted_ps.sum(1) - 1).abs()])
    assert weighted_ps.size(0) == ps.size(0)
    assert weighted_ps.size(-1) == ps.size(-1)
    assert len(weighted_ps.size()) == 2

    return weighted_ps

def to_bins(reward, num_bins):
    """
    Turns a float reward value between 0 and 1 into a histogram
    :param reward:  float
    :param num_bins: long
    :return: num_bins floats
    """
    out = torch.zeros(num_bins, device=reward.device, dtype=reward.dtype)
    reward = max(0, min(reward.item(), 1))
    ind = math.floor(reward*(num_bins-1))
    out[ind] = 1
    return out



