import logging
import random
import numpy as np
import os
from unittest import TestCase
from generative_playground.codec.hypergraph import HyperGraph
from generative_playground.codec.hypergraph_grammar import GrammarInitializer
from generative_playground.models.embedder.graph_embedder import GraphEmbedder
from generative_playground.molecules.data_utils.zinc_utils import get_zinc_molecules
from generative_playground.models.decoder.graph_decoder import GraphEncoder
from generative_playground.codec.codec import get_codec


# create a grammar from scratch # TODO: later, want to load a cached grammar instead
tmp_file = 'tmp.pickle'
# delete the cached files
if os.path.isfile(tmp_file):
    os.remove(tmp_file)
if os.path.isfile('init_' + tmp_file):
    os.remove('init_' + tmp_file)

gi = GrammarInitializer(tmp_file)


# run a first run for 10 molecules
first_10 = gi.init_grammar(10)

class TestGraphEmbedder(TestCase):
    def test_graph_embedder_on_complete_hypergraphs(self):
        ge = GraphEmbedder(target_dim=512, grammar=gi.grammar)
        mol_graphs = [HyperGraph.from_mol(mol) for mol in get_zinc_molecules(5)]
        out = ge(mol_graphs)

    # TODO: fix this test
    # def test_graph_embedder_on_nx_graphs(self):
    #     ge = GraphEmbedder(100, gi.grammar)
    #     mol_graphs = [HyperGraph.from_mol(mol) for mol in get_zinc_molecules(5)]
    #     out = ge([mg.to_nx() for mg in mol_graphs])

    def test_graph_embedder_on_hgs_with_nonterminals(self):
        ge = GraphEmbedder(target_dim=512, grammar=gi.grammar)
        graphs = gi.grammar.rules[1:11] # the first rule is None, corresponding to the padding index
        out = ge(graphs)


    def test_graph_encoder(self):
        encoder = GraphEncoder(grammar=gi.grammar,
                               d_model=512,
                               drop_rate=0.0)
        mol_graphs = [HyperGraph.from_mol(mol) for mol in get_zinc_molecules(5)]
        out = encoder(mol_graphs)
        print('success!')

