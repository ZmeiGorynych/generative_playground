from generative_playground.codec.hypergraph_grammar import HypergraphGrammar
import logging
import random
import numpy as np
import os
import networkx as nx
from unittest import TestCase

class TestStart(TestCase):
    def test_cycle_length(self):
        g = HypergraphGrammar.load('hyper_grammar.pickle')
        for r in g.rules:
            if r is not None:
                g = r.to_nx()
                cycles = nx.minimum_cycle_basis(g)
                if len(cycles) > 0:
                    maxlen = max([len(c) for c in cycles])
                    if maxlen > 7:
                        print(maxlen)

                cc = nx.number_connected_components(g)
                if cc>1:
                    print(cc)