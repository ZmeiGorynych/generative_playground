import numpy as np
from unittest import TestCase
import torch
from generative_playground.utils.gpu_utils import device
from generative_playground.models.losses.wasserstein_loss import *
eps = 1e-5

class TestStart(TestCase):
    def test_normalize_distribution(self):
        e1 = torch.Tensor([[2., 0., 0.],
                           [0., 3., 0.],])
        e1n = normalize_distribution(e1)
        assert e1n[0,0] == 1.0
        assert e1n[0].sum() == 1.0

    def test_wasserstein_matrix_2d(self):
        out = wasserstein_matrix(2)
        assert (out - torch.Tensor([[1., 1.],
                           [0., 1.]])).abs().sum() < eps

    def test_wasserstein_basis_decomp(self):
        M = wasserstein_matrix(3)
        test = torch.tensor([[-1.,1.,0.]]).transpose(0,1)
        assert (torch.mm(M, test).view(-1)- torch.tensor([0.,1.,0.])).abs().sum() < eps

    def test_wasserstein_einsum_decomp(self):
        M = wasserstein_matrix(3)
        test = torch.tensor([[-1., 1., 0.]])
        out = torch.einsum("mn,bn->bm", (M, test))
        assert (out.view(-1) - torch.tensor([0., 1., 0.])).abs().sum() < eps

    def test_wasserstein_einsum_decomp_more(self):
        M = wasserstein_matrix(3)
        test = torch.tensor([[-1.,0.,1.]])
        out = torch.einsum("mn,bn->bm", (M, test))
        assert (out.view(-1) - torch.tensor([0.,1.,1.])).abs().sum() < eps

    def test_unit_vector_distance(self):
        # test distance between e_i and e_j
        e1 = torch.Tensor([[1.,0.,0.,0.]])
        e2 = torch.Tensor([[0.,1.,0.,0.]])
        e3 = torch.Tensor([[0.,0.,0.,1.]])
        w = WassersteinLoss()
        assert w(e1,e2) == 1
        assert w(e1,e3) == 3


    def test_symmetry(self):
        x1 = torch.rand(1,5)
        x2 = torch.rand(1,5)
        w = WassersteinLoss()
        assert w(x1, x2) == w(x2, x1)

    def test_differentiable(self):
        x1 = torch.rand(1,5)
        x2 = torch.rand(1,5)
        x1.requires_grad = True
        w = WassersteinLoss()
        thisw = w(x1, x2)
        thisw.backward()
