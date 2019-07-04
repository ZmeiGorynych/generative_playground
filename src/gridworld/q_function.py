import random
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .distribution import DistributionCalculator


class QDistFunction(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.smax = nn.Softmax()
        self.gamma = gamma
        self.distr_calc = DistributionCalculator(
            torch.tensor(list(range(-10, 11)), dtype=torch.float32), gamma
        )
        self.bins = len(self.distr_calc.bin_mids)
        self.l1 = nn.Linear(64, 164)
        self.l2 = nn.Linear(164, 150)
        self.l3 = nn.Linear(150, self.bins * 4)

    def forward(self, x):
        x = x.view(-1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        x = x.view(4, -1)
        return F.softmax(x, dim=-1)

    def fit_step(
        self,
        old_state,
        new_state,
        action,
        reward,
        done,
        gamma,
        loss_fun,
        optimizer,
        range_,
        grad_clip: Optional[float] = None,
    ) -> None:
        self.train()

        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = self(old_state)[action[0], :]
        new_qdist = self(new_state).detach()
        aggregated_dist = self.distr_calc.aggregate_distributions_best_exp_value(
            new_qdist
        )
        new_qdist = self.distr_calc.shift_distribution(aggregated_dist, reward, done)

        # introduce a batch dimension since the thingy seems to be expecting it
        loss = loss_fun(outputs.unsqueeze(0), new_qdist.detach().unsqueeze(0))
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.parameters(),
                grad_clip
            )
        optimizer.step()
        self.eval()

    def select_action(self, state, greedy=False):
        qdist = self(
            torch.tensor(state, dtype=torch.float32)
        )
        if greedy:
            exp_values = (qdist * self.distr_calc.bin_mids).sum(1)
            return torch.argmax(exp_values)
        samples = torch.multinomial(qdist, 1)
        return samples.max(0)[1]


class QFunction(nn.Module):
    def __init__(self, epsilon=0.1, gamma=0.9):
        super().__init__()
        self.l1 = nn.Linear(64, 164)
        self.l2 = nn.Linear(164, 150)
        self.l3 = nn.Linear(150, 4)
        self.epsilon = epsilon
        self.gamma = gamma

    def forward(self, x):
        x = x.view(-1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3(x)

    def fit_step(
        self,
        old_state,
        state,
        action,
        reward,
        done,
        gamma,
        loss_fun,
        optimizer,
        range_=None,
        grad_clip: Optional[float] = None,
    ) -> None:
        self.train()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = self(old_state)[action]

        new_q = self(
            torch.tensor(state, dtype=torch.float32)
        )
        greedy_q = new_q.max(0)[0]
        if done:
            target = torch.tensor(reward, dtype=torch.float32)
        else:
            target = reward + self.gamma * greedy_q
        target = target.detach()

        loss = loss_fun(outputs, target)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.parameters(),
                grad_clip
            )
        optimizer.step()
        self.eval()

    def select_action(self, state, greedy=False):
        qvals = self(
            torch.tensor(state, dtype=torch.float32)
        )
        if not greedy and random.random() < self.epsilon:
            return random.randint(0, 3)
        return qvals.max(0)[1]
