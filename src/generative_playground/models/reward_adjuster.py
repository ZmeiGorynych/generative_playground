from collections import defaultdict
import math

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
