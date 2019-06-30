import torch.nn as nn
import torch


class WassersteinLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.M = None

    def forward(self, x1, x2):
        """
        Calculates the Wasserstein loss between two discrete distributions
        :param x1: batch x bin floats
        :param x2: batch x bin floats
        :return: mean Wasserstein distance across the batch
        """
        assert all([x1.size(i) == x2.size(i) for i in range(len(x1.size()))]), "Shape mismatch in Wasserstein Loss"
        x1_ = normalize_distribution(x1)
        x2_ = normalize_distribution(x2)
        if self.M is None:
            self.M = wasserstein_matrix(x1.size(-1)).to(dtype=x1.dtype, device=x1.device)
        wasserstein_coeffs = torch.einsum("mn,bn->bm", (self.M, x1_ - x2_))
        assert wasserstein_coeffs[:, 0].abs().max() < 1e-5
        loss = wasserstein_coeffs[:, 1:].abs().sum(1).mean()
        return loss


def normalize_distribution(x):
    assert all(x.view(-1)>=0), "Distributions must be non-negative"
    totalx = x.sum(-1, keepdim=True)
    out = x / totalx
    if len(out.size()) == 1:
        assert (out.sum(-1)-1).abs() < 1e-5, "Normalization fail"
    else:
        assert all((out.sum(-1)-1).abs() < 1e-5), "Normalization fail"
    return out


def wasserstein_matrix(n):
    """
    Returns the basis matrix for shoveling probability between neighboring batches
    First vector is unit vector so the matrix is invertable
    Basis vectors are in columns, so you might need to transpose that before using
    :param n: matrix size
    :return: nxn floats
    """
    out = torch.zeros(n, n, requires_grad=False)
    for i in range(n):
        out[i, i] = 1
        if i > 0:
            out[i - 1, i] = -1

    out = out.inverse()
    return out
