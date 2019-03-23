import logging
import random
import numpy as np
import os
from unittest import TestCase
from generative_playground.codec.hypergraph_grammar import GrammarInitializer

class TestGrammarInitializer(TestCase):
    def test_grammar_initializer(self):
        tmp_file = 'tmp.pickle'
        gi = GrammarInitializer(tmp_file)
        # delete the cached files
        if os.path.isfile(gi.own_filename):
            os.remove(gi.own_filename)
        if os.path.isfile(gi.grammar_filename):
            os.remove(gi.grammar_filename)
        # run a first run for 10 molecules
        first_10 = gi.init_grammar(10)
        # load the resulting object
        gi2 = GrammarInitializer.load(gi.own_filename)
        gi2.init_grammar(20)
