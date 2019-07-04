import torch
import numpy as np
eps = 1e-5


class DistributionCalculator:
    def __init__(self, bin_boundaries, gamma):
        """

        :param bin_boundaries: for now, assume integer bin boundaries
        """
        self.bin_bounds = bin_boundaries
        self.bin_mids = 0.5 * (bin_boundaries[:-1] + bin_boundaries[1:])
        self.gamma = gamma

    def shift_distribution(self, dist, reward, done):
        assert len(dist) + 1 == len(self.bin_bounds)
        m = torch.zeros_like(dist)
        for bin, p in enumerate(dist):
            if done:
                tzj = reward
            else:
                tzj = reward + self.gamma * self.bin_mids[bin]
            tzj = max(tzj, self.bin_mids[0])
            tzj = min(tzj, self.bin_mids[-1])
            bj = tzj - self.bin_bounds[0]
            lower = int(np.floor(bj))
            dp = p * (bj - lower)
            m[min(lower+1, len(m)-1)] += dp
            m[lower] += p - dp
        assert (dist.sum(-1) - m.sum(-1)).abs() < eps
        return m

    def argmax_exp_value(self, distrs):
        assert len(distrs.size()) == 2
        assert distrs.size(1) == len(self.bin_mids)
        exp_values = (distrs*self.bin_mids).sum(1)
        return torch.argmax(exp_values)

    def aggregate_distributions_best_exp_value(self, distrs):
        best_ev = self.argmax_exp_value(distrs)
        return distrs[best_ev, :]
