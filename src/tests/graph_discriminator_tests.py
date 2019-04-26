import logging
import random
import numpy as np
import os
from unittest import TestCase
from generative_playground.codec.hypergraph import HyperGraph
from generative_playground.codec.hypergraph_grammar import GrammarInitializer
from generative_playground.models.embedder.graph_embedder import GraphEmbedder
from generative_playground.molecules.data_utils.zinc_utils import get_zinc_molecules, get_zinc_smiles
from generative_playground.models.decoder.graph_decoder import GraphEncoder
from generative_playground.codec.codec import get_codec
from generative_playground.models.heads.attention_aggregating_head import *
from generative_playground.models.heads.multiple_output_head import MultipleOutputHead
from generative_playground.molecules.models.graph_discriminator import GraphDiscriminator

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

class TestGraphDiscriminator(TestCase):
    def test_with_multihead_attenion_aggregating_head(self):
        d_model = 512
        encoder = GraphEncoder(grammar=gi.grammar,
                               d_model=d_model,
                               drop_rate=0.0)

        encoder_aggregated = MultiheadAttentionAggregatingHead(encoder)
        mol_graphs = [HyperGraph.from_mol(mol) for mol in get_zinc_molecules(5)]
        out = encoder_aggregated(mol_graphs)
        assert out.size(0) == len(mol_graphs)
        assert out.size(1) == d_model
        assert len(out.size()) == 2

    def test_with_first_sequence_element_head(self):
        d_model = 512
        encoder = GraphEncoder(grammar=gi.grammar,
                               d_model=d_model,
                               drop_rate=0.0)

        encoder_aggregated = FirstSequenceElementHead(encoder)
        mol_graphs = [HyperGraph.from_mol(mol) for mol in get_zinc_molecules(5)]
        out = encoder_aggregated(mol_graphs)
        assert out.size(0) == len(mol_graphs)
        assert out.size(1) == d_model
        assert len(out.size()) == 2

    def test_full_discriminator_parts(self):
        encoder = GraphEncoder(grammar=gi.grammar,
                               d_model=512,
                               drop_rate=0.0)

        encoder_aggregated = FirstSequenceElementHead(encoder)
        discriminator = MultipleOutputHead(encoder_aggregated, [1])
        mol_graphs = [HyperGraph.from_mol(mol) for mol in get_zinc_molecules(5)]
        out = discriminator(mol_graphs)
        assert out[0].size(0) == len(mol_graphs)
        assert out[0].size(1) == 1
        assert len(out[0].size()) == 2

    def test_discriminator_class(self):
        d = GraphDiscriminator(gi.grammar, drop_rate=0.1)
        smiles = get_zinc_smiles(5)
        out = d(smiles)
        assert out['p_zinc'].size(0) == len(smiles)
        assert out['p_zinc'].size(1) == 2
        assert len(out['p_zinc'].size()) == 2
        assert out['smiles'] == smiles

    def test_discriminator_class_dict_input(self):
        d = GraphDiscriminator(gi.grammar, drop_rate=0.1)
        smiles = get_zinc_smiles(5)
        out = d({'smiles':smiles, 'test': 'test'})
        assert out['p_zinc'].size(0) == len(smiles)
        assert out['p_zinc'].size(1) == 2
        assert len(out['p_zinc'].size()) == 2
        assert out['smiles'] == smiles
        assert out['test'] == 'test'


