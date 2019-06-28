import numpy as np
from unittest import TestCase
import torch
from generative_playground.utils.gpu_utils import device
from generative_playground.models.discrete_distribution_utils import *
eps = 1e-5

class TestStart(TestCase):
    def test_to_bins_one(self):
        reward = torch.tensor(1.0)
        out = to_bins(reward, 4)
        tgt = torch.tensor([0.,0.,0.,1.])
        assert (tgt-out).abs().sum() < eps

    def test_to_bins_lowest(self):
        reward = torch.tensor(-1.0)
        out = to_bins(reward, 5)
        tgt = torch.tensor([1.,0.,0.,0.,0.])
        assert (tgt-out).abs().sum() < eps

    def test_calc_expected_value(self):
        val = CalcExpectedValue()
        probs = torch.diag(torch.ones(5))
        out = val(probs)
        assert len(out.size()) ==1
        tgt = torch.tensor([0.1250, 0.3750, 0.6250, 0.8750, 1.0000])
        assert (out-tgt).abs().sum()< eps

    def test_softmax_policy_probs(self):
        calc = SoftmaxPolicyProbsFromDistributions()
        probs = torch.diag(torch.ones(5)).unsqueeze(0) # make this into a batch with 5 actions
        out = calc(probs, 1.0)
        assert len(out.size()) == 2
        assert torch.argmax(out[0,:]) == 4

    def test_aggregate_distributions(self):
        probs = F.softmax(torch.rand(3,5,6), dim=-1)
        policy_p = SoftmaxPolicyProbsFromDistributions()(probs, 0.01)
        out = aggregate_distributions_by_policy(probs, policy_p)
        assert out.size(0) == probs.size(0)
        assert out.size(1) == probs.size(2)
        assert len(out.size()) == 2

    def test_thompson_policy_probs(self):
        probs = F.softmax(torch.randn(10, 5, 50), dim=-1)
        mask = torch.ones(10,5)
        mask[:,3] = 0 # let's mask one prob out
        mask = mask > 0
        thompson_probs = thompson_probabilities(probs, mask)
        assert len(thompson_probs.size()) == 2