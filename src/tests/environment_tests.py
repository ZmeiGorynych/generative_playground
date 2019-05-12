import torch
import random
import numpy as np
import os
from unittest import TestCase
from generative_playground.codec.hypergraph_grammar import GrammarInitializer
from generative_playground.models.decoder.policy import SoftmaxRandomSamplePolicy
from generative_playground.utils.gpu_utils import device
from generative_playground.codec.hypergraph_grammar import evaluate_rules, HypergraphGrammar, HypergraphMaskGenerator
from generative_playground.models.problem.rl.environment import GraphEnvironment


class Environment(TestCase):
    def test_hypergraph_mask_gen_step(self):
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
        next_action = (None, [None for _ in range(batch_size)])
        while True:
            try:
                graphs, node_mask, full_logit_priors = mask_gen.step(next_action)
                next_node = np.argmax(node_mask, axis=1)
                next_action_ = [np.argmax(full_logit_priors[b, next_node[b]]) for b in range(batch_size)]
                next_action = (next_node, next_action_)
            except StopIteration:
                break

        # def test_hypergraph_mask_gen_unconditional_priors(self):
        #     tmp_file = 'tmp2.pickle'
        #     gi = GrammarInitializer(tmp_file)
        #     gi.delete_cache()
        #     # now create a clean new one
        #     gi = GrammarInitializer(tmp_file)
        #     # run a first run for 10 molecules
        #     gi.init_grammar(20)
        #     gi.grammar.check_attributes()
        #     mask_gen = HypergraphMaskGenerator(30, gi.grammar, priors=True)
        #     env = GraphEnvironment(mask_gen,
        #                            reward_fun=lambda x: np.zeros(len(x)),
        #                            batch_size=2)
        #
        #     next_action = [None for _ in range(env.batch_size)]
        #     while True:
        #         try:
        #             next_state, reward, done, (self.smiles, self.valid) = env.step(next_action)
        #             graphs, node_mask, full_logit_priors = next_state
        #             next_node = np.argmax(node_mask, dim=1)
        #             next_action = [np.argmax(full_logit_priors[b, next_node[b]]) for b in range(env.batch_size)]
        #         except StopIteration:
        #             break