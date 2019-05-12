import torch
import random
import numpy as np
import os
from unittest import TestCase
from generative_playground.codec.hypergraph_grammar import GrammarInitializer
from generative_playground.models.decoder.decoders import DecoderWithEnvironmentNew
from generative_playground.utils.gpu_utils import device
from generative_playground.codec.hypergraph_grammar import evaluate_rules, HypergraphGrammar, HypergraphMaskGenerator
from generative_playground.models.problem.rl.environment import GraphEnvironment
from generative_playground.models.decoder.graph_decoder import GraphEncoder, GraphDecoderWithNodeSelection
from generative_playground.models.heads.multiple_output_head import MultipleOutputHead
from generative_playground.models.losses.policy_gradient_loss import PolicyGradientLoss

class Environment(TestCase):
    def test_decoder_with_environment_new(self):
        tmp_file = 'tmp2.pickle'
        gi = GrammarInitializer(tmp_file)
        gi.delete_cache()
        # now create a clean new one
        gi = GrammarInitializer(tmp_file)
        # run a first run for 10 molecules
        gi.init_grammar(20)
        gi.grammar.check_attributes()
        mask_gen = HypergraphMaskGenerator(30, gi.grammar, priors=True)
        batch_size = 2

        env = GraphEnvironment(mask_gen,
                               reward_fun=lambda x: 2 * np.ones(len(x)),
                               batch_size=2)
        encoder = GraphEncoder(grammar=gi.grammar,
                               d_model=512,
                               drop_rate=0.0,
                               model_type='transformer')
        model = MultipleOutputHead(model=encoder,
                                   output_spec={'node': 1,  # to be used to select next node to expand
                                                'action': len(gi.grammar)},  # to select the action for chosen node
                                   drop_rate=0.0)
        # don't support using this model in VAE-style models yet
        model.init_encoder_output = lambda x: None
        stepper = GraphDecoderWithNodeSelection(model)

        decoder = DecoderWithEnvironmentNew(stepper, env)
        out = decoder()
        loss = PolicyGradientLoss(loss_type='advantage_record_mean_best')
        this_loss = loss(out)
        this_loss.backward()
        print('done!')
