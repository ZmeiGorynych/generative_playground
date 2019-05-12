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


class Environment(TestCase):
    def test_graph_environment_step(self):
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
                               reward_fun=lambda x: np.zeros(len(x)),
                               batch_size=2)
        graphs, node_mask, full_logit_priors = env.reset()
        while True:
            try:
                next_node = np.argmax(node_mask, axis=1)
                next_action_ = [np.argmax(full_logit_priors[b, next_node[b]]) for b in range(batch_size)]
                next_action = (next_node, next_action_)
                (graphs, node_mask, full_logit_priors), reward, done, info = env.step(next_action)
            except StopIteration:
                break

        print(info)

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
                               reward_fun=lambda x: 2*np.ones(len(x)),
                               batch_size=2)

        def dummy_stepper(state):
            graphs, node_mask, full_logit_priors = state
            next_node = np.argmax(node_mask, axis=1)
            next_action_ = [np.argmax(full_logit_priors[b, next_node[b]]) for b in range(batch_size)]
            next_action = (next_node, next_action_)
            return next_action, np.zeros(len(state))

        dummy_stepper.output_shape = [None, None, None]
        dummy_stepper.init_encoder_output = lambda x: None

        decoder = DecoderWithEnvironmentNew(dummy_stepper, env, batch_size=batch_size)
        out = decoder()
        print('done!')

