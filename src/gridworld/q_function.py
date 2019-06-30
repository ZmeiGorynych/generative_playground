from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dist_utils import to_bins


class QDistFunction(nn.Module):
    def __init__(self, bins):
        super().__init__()
        self.bins = bins
        self.l1 = nn.Linear(64, 164)
        self.l2 = nn.Linear(164, 150)
        self.l3 = nn.Linear(150, bins * 4)
        self.smax = nn.Softmax()

    def forward(self, x):
        x = x.view(-1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        # view -> softmax
        x = x.view(4, -1)
        return F.softmax(x, dim=-1)

    def fit_step(
        self,
        old_state,
        new_state,
        action,
        reward,
        range_,
        gamma,
        loss_fun,
        optimizer,
            distr_calc,
        grad_clip: Optional[float] = None,
    ) -> None:
        self.train()

        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = self(old_state)[action[0], :]
        new_qdist = self(new_state).detach()
        aggregated_dist = distr_calc.aggregate_distributions_best_exp_value(new_qdist)
        new_qdist = distr_calc.shift_distribution(aggregated_dist, reward)

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
