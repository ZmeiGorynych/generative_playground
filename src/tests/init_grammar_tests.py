import logging
import random
import numpy as np
import os
from unittest import TestCase
from generative_playground.codec.hypergraph_grammar import GrammarInitializer

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
            cond_count += sum(cf.values())
        count = sum(gi2.grammar.rule_frequency_dict.values())
        assert cond_count == count, "Something went wrong when counting the frequencies..."
