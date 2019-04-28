from unittest import TestCase
from generative_playground.models.dummy import DummyModel

class TestStart(TestCase):
    def test_ones(self):
        output_shape = (10, 15, 20)
        d = DummyModel(output_shape, False)
        out = d()
        assert tuple(out.size()) == output_shape

    def test_random(self):
        output_shape = (10, 15, 20)
        d = DummyModel(output_shape, True)
        out = d()
        assert tuple(out.size()) == output_shape
