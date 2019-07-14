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

    def test_graph_conditional_decoder(self):
        decoder_type = 'graph_conditional'
        grammar = 'hypergraph:' + tmp_file
        print("testing ", decoder_type, grammar)
        out = self.generic_decoder_test(decoder_type, grammar)
        print('success!')
