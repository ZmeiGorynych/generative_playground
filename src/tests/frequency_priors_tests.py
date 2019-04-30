import logging
import random
import numpy as np
import torch
import os
from unittest import TestCase
from generative_playground.codec.hypergraph_grammar import evaluate_rules, HypergraphGrammar, HypergraphMaskGenerator
from generative_playground.molecules.data_utils.zinc_utils import get_zinc_smiles
from generative_playground.codec.codec import get_codec
from generative_playground.models.decoder.policy import SoftmaxRandomSamplePolicy
from generative_playground.utils.gpu_utils import device


class TestStart(TestCase):
    def test_hypergraph_mask_gen(self):
        molecules = True
        grammar_cache = 'tmp.pickle'
        grammar = 'hypergraph:' + grammar_cache
        # create a grammar cache inferred from our sample molecules
        g = HypergraphGrammar(cache_file=grammar_cache)
        if os.path.isfile(g.cache_file):
            os.remove(g.cache_file)
        g.strings_to_actions(get_zinc_smiles(5))
        mask_gen1 = get_codec(molecules, grammar, 30).mask_gen
        mask_gen2 = get_codec(molecules, grammar, 30).mask_gen
        mask_gen1.priors = False
        mask_gen2.priors = True
        policy1 = SoftmaxRandomSamplePolicy(bias=mask_gen1.grammar.get_log_frequencies())
        policy2 = SoftmaxRandomSamplePolicy()
        lp = []
        for mg in [mask_gen1,mask_gen2]:
            mg.reset()
            mg.apply_action([None])
            logit_priors = mg.action_prior_logits()  # that includes any priors
            lp.append(torch.from_numpy(logit_priors).to(device=device, dtype=torch.float32))

        dummy_model_output = torch.ones_like(lp[0])
        eff_logits = []
        for this_lp, policy in zip(lp, [policy1, policy2]):
            eff_logits.append(policy.effective_logits(dummy_model_output))

        assert torch.max((eff_logits[0] -eff_logits[1]).abs()) < 1e-6
