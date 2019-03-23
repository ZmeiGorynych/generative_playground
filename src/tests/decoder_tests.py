import logging
import random
import numpy as np
import os
import torch
from unittest import TestCase
from generative_playground.models.decoder.decoder import get_decoder
from generative_playground.codec.codec import get_codec
from generative_playground.codec.hypergraph_grammar import GrammarInitializer

# make sure there's a cached grammar for us to use
tmp_file = 'tmp.pickle'
if not os.path.isfile(tmp_file):
    gi = GrammarInitializer(tmp_file)
    gi.init_grammar(10)

z_size = 200
batch_size = 5
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
        all_sum = torch.sum(out[1])
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

    # TODO: action_resnet is broken, maybe want to fix some day