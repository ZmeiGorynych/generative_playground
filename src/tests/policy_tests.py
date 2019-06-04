import numpy as np
from unittest import TestCase
import torch
from generative_playground.utils.gpu_utils import device
from generative_playground.models.problem.policy import SoftmaxRandomSamplePolicy, MaxPolicy, PolicyFromTarget

logits = torch.Tensor([[-10., 1., -10.],[-10., 1., -10.]]).to(device=device, dtype=torch.float32)

class TestStart(TestCase):
    def test_softmax_random_policy(self):
        policy = SoftmaxRandomSamplePolicy()
        ind = policy(logits)
        assert len(ind) == 2

    def test_softmax_random_policy_with_bias(self):
        policy = SoftmaxRandomSamplePolicy(bias=np.array([10, 0, 0]))
        ind = policy(logits)
        assert len(ind) == 2

    def test_softmax_random_policy_with_list_bias(self):
        policy = SoftmaxRandomSamplePolicy(bias=[10, 0, 0])
        ind = policy(logits)
        assert len(ind) == 2

    def test_max_policy(self):
        policy = MaxPolicy()
        ind = policy(logits)
        assert ind[0] == 1 and ind[1] == 1

    def test_target_policy_numpy_target(self):
        policy = PolicyFromTarget(np.array([[1, 2], [0, 1]]))
        ind1 = policy(logits)
        assert ind1[0] == 1 and ind1[1] == 0
        ind2 = policy(logits)
        assert ind2[0] == 2 and ind2[1] == 1
        self.assertRaises(StopIteration, lambda: policy(logits))

    def test_target_policy_pytorch_target(self):
        policy = PolicyFromTarget(torch.Tensor([[1, 2], [0, 1]]))
        ind1 = policy(logits)
        assert ind1[0] == 1 and ind1[1] == 0
        ind2 = policy(logits)
        assert ind2[0] == 2 and ind2[1] == 1
        self.assertRaises(StopIteration, lambda: policy(logits))




