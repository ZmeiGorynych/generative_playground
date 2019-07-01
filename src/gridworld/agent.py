import numpy as np
import pandas as pd
import torch
from tqdm import tqdm_notebook as tqdm

from .dist_utils import display_values
from .q_function import QDistFunction


class Agent:
    def __init__(
        self,
        env,
        epsilon,
        loss_fun,
        q_function,
        gamma,
        lr=0.01,
        max_steps=100,
    ):
        self.q = q_function
        self.env = env
        self.epsilon = epsilon
        self.gamma = gamma
        self.max_steps = max_steps
        self.loss_fun = loss_fun
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr)
        self.range_ = list(range(-10, 11))

    def run_episode(self):
        done = False
        total_reward = 0.0
        while not done:
            if isinstance(self.q, QDistFunction):
                display_values(self.env, self.q, self.range_)
            action = self.q.select_action(self.env.state)

            old_state, reward, _, done = self.env.step(action)
            total_reward += reward

            self.q.fit_step(
                torch.tensor(old_state, dtype=torch.float32),
                torch.tensor(self.env.state, dtype=torch.float32),
                action,
                reward,
                done,
                self.gamma,
                self.loss_fun,
                self.optimizer,
                self.range_,
            )
        return total_reward

    def run_model(self, world=0):
        done = False
        self.env.reset()
        print(self.env.display())
        for _ in range(self.max_steps):
            action = self.q.select_action(self.env.state, greedy=True)
            _, _, _, done = self.env.step(action)
            print(self.env.display())
            if done:
                break

    def train(self, epochs=1000):
        self.env.reset()
        rewards = np.zeros(epochs)
        for i in tqdm(range(epochs)):
            rewards[i] = self.run_episode()
            self.env.reset()
        return pd.Series(rewards)
