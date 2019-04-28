from unittest import TestCase
import torch
from generative_playground.utils.gpu_utils import device
from generative_playground.models.dummy import DummyModel, PassthroughModel
from generative_playground.models.heads.multiple_output_head import MultipleOutputHead

class TestMOHead(TestCase):
    def test_batch_crosstalk(self):
        output_shape = [10,15,20]
        m = PassthroughModel(output_shape)
        h = MultipleOutputHead(m, [2]).to(device)
        inp = torch.ones(*output_shape).to(device)
        out1 = h(inp)[0]
        out2 = h(inp[:1])[0]
        assert torch.max((out1[:1]-out2).abs()) < 1e-6
