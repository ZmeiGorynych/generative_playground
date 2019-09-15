import logging
import random
import numpy as np
import os
from unittest import TestCase
from generative_playground.codec.hypergraph_grammar import GrammarInitializer, normalize_frequencies

class TestGrammarInitializer(TestCase):
    def test_grammar_initializer(self):
        # nuke any cached data
        tmp_file = 'tmp2.pickle'
        gi = GrammarInitializer(tmp_file)
        gi.delete_cache()
        # now create a clean new one
        gi = GrammarInitializer(tmp_file)
        # run a first run for 10 molecules
        first_10 = gi.init_grammar(10)
        # load the resulting object
        gi2 = GrammarInitializer.load(gi.own_filename)
        gi2.init_grammar(20)

        freqs = gi2.grammar.get_log_frequencies()
        assert len(freqs) == len(gi2.grammar)
        assert all([f >= 0 for f in freqs])
        cond_count = 0
        for cf in gi2.grammar.conditional_frequencies.values():
            this_count = sum(cf.values())
            assert abs(this_count-1.0) < 1e-5
            cond_count += sum(cf.values())
        nt_count = 0
        for rule in gi2.grammar.rules:
            if rule is not None:
                nt_count += len(rule.nonterminal_ids())
        assert cond_count == nt_count, "Something went wrong when counting the frequencies..."
        gi2.grammar.check_attributes()

    def test_normalize_frequencies(self):
        x = {x:x for x in range(5)}
        out = normalize_frequencies(x)
        assert out[1] == 0.1

