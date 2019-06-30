import math
import torch


def to_bins(reward, num_bins, min_=-10, max_=10):
    """
    Turns a float reward into a histogram
    :param reward:  float
    :param num_bins: long
    :return: num_bins floats
    """
    out = torch.zeros(num_bins, device=reward.device, dtype=reward.dtype)
    reward = max(min_, min(reward.item(), max_))
    ind = math.floor(reward)
    out[ind] = 1
    return out
