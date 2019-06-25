import logging
import random
import numpy as np
import os
import torch
from unittest import TestCase
from generative_playground.models.decoder.decoder import get_decoder
from generative_playground.models.decoder.graph_decoder import GraphEncoder
from generative_playground.codec.codec import get_codec
from generative_playground.codec.hypergraph_grammar import GrammarInitializer
from generative_playground.models.heads import MultipleOutputHead
from generative_playground.codec.hypergraph import HyperGraph
from generative_playground.molecules.data_utils.zinc_utils import get_zinc_molecules
from generative_playground.utils.gpu_utils import device

# make sure there's a cached grammar for us to use
tmp_file = 'tmp.pickle'
if os.path.isfile(tmp_file):
    os.remove(tmp_file)
if os.path.isfile('init_' + tmp_file):
    os.remove('init_' + tmp_file)

gi = GrammarInitializer(tmp_file)
gi.init_grammar(10)

z_size = 200
batch_size = 2
max_seq_length = 30

class TestDecoders(TestCase):
    def generic_decoder_test(self, decoder_type, grammar):
        codec = get_codec(molecules=True, grammar=grammar, max_seq_length=max_seq_length)
        decoder, pre_decoder = get_decoder(decoder_type=decoder_type,
                                           max_seq_length=max_seq_length,
                                           grammar=grammar,
                                           feature_len=codec.feature_len(),
                                           z_size=z_size,
                                           batch_size=batch_size)
        out = decoder()
        # it returns all sorts of things: out_actions_all, out_logits_all, out_rewards_all, out_terminals_all, (info[0], to_pytorch(info[1]))
        if 'logits' in out:
            all_sum = torch.sum(out['logits'])
        else:
            all_sum = torch.sum(out['logp'])
        all_sum.backward()
        return all_sum

    def test_random_decoder_all_grammars(self):
        decoder_type = 'random'  # 'action_resnet' fails, maybe want to fix later
        grammars = ['classic', 'new', 'hypergraph:' + tmp_file, False]
        for g in grammars:
            print("testing ", decoder_type, g)
            out = self.generic_decoder_test(decoder_type, g)

    def test_rnn_decoder_all_grammars(self):
        decoder_type = 'step'  # 'action_resnet' fails, maybe want to fix later
        grammars = ['classic', 'new', 'hypergraph:' + tmp_file, False]
        for g in grammars:
            print("testing ", decoder_type, g)
            out = self.generic_decoder_test(decoder_type, g)

    def test_rnn_action_decoder_all_grammars(self):
        decoder_type = 'action'  # 'action_resnet' fails, maybe want to fix later
        grammars = ['classic', 'new', 'hypergraph:' + tmp_file, False]
        for g in grammars:
            print("testing ", decoder_type, g)
            out = self.generic_decoder_test(decoder_type, g)

    def test_attention_decoder_all_grammars(self):
        decoder_type = 'attention'  # 'action_resnet' fails, maybe want to fix later
        grammars = ['classic', 'new', 'hypergraph:' + tmp_file, False]
        for g in grammars:
            print("testing ", decoder_type, g)
            out = self.generic_decoder_test(decoder_type, g)

    def test_graph_encoder_with_head(self):
        codec = get_codec(molecules=True,
                          grammar='hypergraph:' + tmp_file,
                          max_seq_length=max_seq_length)
        encoder = GraphEncoder(grammar=gi.grammar,
                               d_model=512,
                               drop_rate=0.0)
        mol_graphs = [HyperGraph.from_mol(mol) for mol in get_zinc_molecules(5)]
        model = MultipleOutputHead(model=encoder,
                                   output_spec={'node': 1,  # to be used to select next node to expand
                                                'action': codec.feature_len()},  # to select the action for chosen node
                                   drop_rate=0.1).to(device)
        out = model(mol_graphs)

    def test_graph_transfomer_decoder(self):
        decoder_type = 'attn_graph'
        grammar = 'hypergraph:' + tmp_file
        print("testing ", decoder_type, grammar)
        out = self.generic_decoder_test(decoder_type, grammar)
        print('success!')

    def test_graph_rnn_decoder(self):
        decoder_type = 'rnn_graph'
        grammar = 'hypergraph:' + tmp_file
        print("testing ", decoder_type, grammar)
        out = self.generic_decoder_test(decoder_type, grammar)
        print('success!')

    def test_graph_transformer_node_decoder(self):
        decoder_type = 'attn_graph_node'
        grammar = 'hypergraph:' + tmp_file
        print("testing ", decoder_type, grammar)
        out = self.generic_decoder_test(decoder_type, grammar)
        print('success!')

    def test_graph_rnn_node_decoder(self):
        decoder_type = 'rnn_graph_node'
        grammar = 'hypergraph:' + tmp_file
        print("testing ", decoder_type, grammar)
        out = self.generic_decoder_test(decoder_type, grammar)
        print('success!')

    def test_graph_transformer_distr_decoder(self):
        decoder_type = 'attn_graph_distr'
        grammar = 'hypergraph:' + tmp_file
        print("testing ", decoder_type, grammar)
        out = self.generic_decoder_test(decoder_type, grammar)
        print('success!')

    def test_graph_rnn_distr_decoder(self):
        decoder_type = 'rnn_graph_distr'
        grammar = 'hypergraph:' + tmp_file
        print("testing ", decoder_type, grammar)
        out = self.generic_decoder_test(decoder_type, grammar)
        print('success!')

    # TODO: 'action_resnet' decoder is broken, maybe want to fix some day