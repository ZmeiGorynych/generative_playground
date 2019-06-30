import math

import numpy as np
import torch

from .environment import Gridworld


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


def display_values(env, q, range_, env_creator=Gridworld.deterministic_easy):
    values = np.zeros((env.height, env.width))
    greedy_actions = np.empty((env.height, env.width), dtype=str)
    actions = ('up', 'right', 'down', 'left',)
    for i in range(env.height):
        for j in range(env.width):
            dummy_env = env_creator()
            dummy_env.set_player_position((i, j))
            outputs = q(
                torch.tensor(dummy_env.state, dtype=torch.float32)
            )
            evs = [
                sum(z * p for z, p in zip(range_, dist))
                for dist in outputs
            ]
            a = np.argmax(evs)
            values[i][j] = evs[a]
            greedy_actions[i][j] = actions[a]
    print(values)
    return values, greedy_actions
