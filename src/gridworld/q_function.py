from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dist_utils import to_bins


class QDistFunction(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(64, 164)
        self.l2 = nn.Linear(164, 150)
        self.l3 = nn.Linear(150, 21 * 4)
        self.smax = nn.Softmax()

    def forward(self, x):
        x = x.view(-1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        # view -> softmax
        x = x.view(4, -1)
        return F.softmax(x, dim=1)

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
        grad_clip: Optional[float] = None,
    ) -> None:
        self.train()

        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = self(old_state)

        new_qdist = self(new_state)
        new_qdist.detach()
        # Softmax this rather than using max EV
        evs = [
            sum(z * p for z, p in zip(range_, dist))
            for dist in outputs
        ]
        a = np.argmax(evs)
        dist = new_qdist[a]
        m = torch.zeros((21,))
        for i, z in enumerate(range_):
            tzj = reward + gamma * z
            if tzj < -10:
                tzj = -10
            elif tzj > 10:
                tzj = 10
            bj = tzj + 10
            l = int(np.floor(bj))
            u = int(np.ceil(bj))
            m[l] += dist[i] * (u - bj)
            m[u] += dist[i] * (bj - l)
        new_qdist[a] = m

        # loss = loss_fun(outputs, new_qdist.detach())
        target = to_bins(torch.tensor(reward), 21)
        new_qdist[a] = target
        loss = loss_fun(outputs, new_qdist.detach())
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.parameters(),
                grad_clip
            )
        optimizer.step()
        self.eval()
