import numpy as np
import pandas as pd
import torch
from tqdm import tqdm_notebook as tqdm

from .q_function import QDistFunction
from generative_playground.models.losses.wasserstein_loss import WassersteinLoss


class Agent:
    def __init__(self, env, epsilon, gamma, lr=0.01, max_steps=100):
        self.q = QDistFunction()
        self.env = env
        self.epsilon = epsilon
        self.gamma = gamma
        self.max_steps = max_steps
        self.loss_fun = WassersteinLoss()
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr)
        self.range_ = list(range(-10, 11))

    def run_episode(self):
        done = False
        total_reward = 0.0
        while not done:
            qdist = self.q(
                torch.tensor(self.env.state, dtype=torch.float32)
            )
            samples = torch.multinomial(qdist, 1)
            print(samples)
            action = samples.max(0)[1]

            old_state, reward, _, done = self.env.step(action)
            total_reward += reward

            self.q.fit_step(
                torch.tensor(old_state, dtype=torch.float32),
                torch.tensor(self.env.state, dtype=torch.float32),
                action,
                reward,
                self.range_,
                self.gamma,
                self.loss_fun,
                self.optimizer
            )
        return total_reward

    def run_model(self, world=0):
        done = False
        self.env.reset()
        print(self.env.display())
        for _ in range(self.max_steps):
            qdist = self.q(torch.tensor(self.env.state, dtype=torch.float32))
            len_range = len(self.range_)
            evs = [
                sum(
                    z * p for z, p in zip(
                        self.range_, qdist[len_range * a:len_range * (a + 1)]
                    )
                )
                for a in range(4)
            ]
            action = np.argmax(evs)

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
