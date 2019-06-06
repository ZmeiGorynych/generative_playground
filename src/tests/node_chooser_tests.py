from unittest import TestCase
from generative_playground.models.losses.policy_gradient_loss import PolicyGradientLoss
from generative_playground.utils.testing_utils import make_grammar, make_decoder
import numpy as np
grammar = make_grammar()


class TestEnvironment(TestCase):
    def test_decoder_with_environment_new(self):

        decoder = make_decoder(grammar, output_spec={'node': 1,  # to be used to select next node to expand
                                       'action': len(grammar)})
        out = decoder()
        loss = PolicyGradientLoss(loss_type='advantage_record_mean_best')
        this_loss = loss(out)
        this_loss.backward()
        print('done!')


