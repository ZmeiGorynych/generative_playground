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
        for eg, g in zip(out,mol_graphs):
            for i in range(len(g), max([len(gg) for gg in mol_graphs])):
                assert eg[i].abs().max() == 0 # embedded values should only be nonzero for actual nodes

    # TODO: fix this test
    # def test_graph_embedder_on_nx_graphs(self):
    #     ge = GraphEmbedder(100, gi.grammar)
    #     mol_graphs = [HyperGraph.from_mol(mol) for mol in get_zinc_molecules(5)]
    #     out = ge([mg.to_nx() for mg in mol_graphs])

    def test_graph_embedder_on_hgs_with_nonterminals(self):
        ge = GraphEmbedder(target_dim=512, grammar=gi.grammar)
        graphs = gi.grammar.rules[1:11] # the first rule is None, corresponding to the padding index
        out = ge(graphs)
        out2 = ge(graphs)
        assert (out - out2).abs().max() < 1e-6, "Embedder should be deterministic!"

    def test_graph_encoder_determinism_transformer(self):
        encoder = GraphEncoder(grammar=gi.grammar,
                               d_model=512,
                               drop_rate=0.0,
                               model_type='transformer')
        mol_graphs = [HyperGraph.from_mol(mol) for mol in get_zinc_molecules(5)]
        out = encoder(mol_graphs)
        out2 = encoder(mol_graphs)
        assert (out - out2).abs().max() < 1e-6, "Encoder should be deterministic with zero dropout!"

    def test_graph_encoder_batch_independence_transformer(self):
        encoder = GraphEncoder(grammar=gi.grammar,
                               d_model=512,
                               drop_rate=0.0,
                               model_type='transformer')
        mol_graphs = [HyperGraph.from_mol(mol) for mol in get_zinc_molecules(5)]
        out = encoder(mol_graphs)
        out2 = encoder(mol_graphs[:1])

        assert (out[:1, :out2.size(1)] - out2).abs().max() < 1e-5, "Encoder should have no crosstalk between batches"

    def test_graph_encoder_determinism_rnn(self):
        encoder = GraphEncoder(grammar=gi.grammar,
                               d_model=512,
                               drop_rate=0.0,
                               model_type='rnn')
        mol_graphs = [HyperGraph.from_mol(mol) for mol in get_zinc_molecules(5)]
        out = encoder(mol_graphs)
        out2 = encoder(mol_graphs)
        assert (out - out2).abs().max() < 1e-6, "Encoder should be deterministic with zero dropout!"

    def test_graph_encoder_batch_independence_rnn(self):
        encoder = GraphEncoder(grammar=gi.grammar,
                               d_model=512,
                               drop_rate=0.0,
                               model_type='rnn')
        mol_graphs = [HyperGraph.from_mol(mol) for mol in get_zinc_molecules(5)]
        out = encoder(mol_graphs)
        out2 = encoder(mol_graphs[:1])

        assert (out[:1,:out2.size(1)] - out2).abs().max() < 1e-5, "Encoder should have no crosstalk between batches"


