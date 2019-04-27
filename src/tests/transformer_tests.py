import logging
import random
import numpy as np
import os
from unittest import TestCase
import torch
from generative_playground.models.transformer.Layers import EncoderLayer
from generative_playground.models.transformer.SubLayers import MultiHeadAttention, \
    PositionwiseFeedForward

d_model = 512
max_steps = 20
class TestGraphDiscriminator(TestCase):
    def test_encoder_layer_determinism(self):
        enc = EncoderLayer(d_model=d_model,
                         d_inner_hid=2*d_model,
                         n_head=6,
                         d_k=128,
                         d_v=128,
                         dropout=0.0,
                         n_max_seq=max_steps,
                         use_attentions=False)
        enc_input = torch.ones(10, max_steps, d_model)
        out1 = enc(enc_input)
        out2 = enc(enc_input)
        assert torch.max((out1[0]-out2[0]).abs()) < 1e-6

    def test_multihead_attention_determinism(self):
        slf_attn = MultiHeadAttention(d_model=d_model,
                         n_head=6,
                         d_k=128,
                         d_v=128,
                         dropout=0.0)

        enc_input = torch.ones(10, max_steps, d_model)
        out1 = slf_attn(enc_input, enc_input, enc_input)
        out2 = slf_attn(enc_input, enc_input, enc_input)
        assert torch.max((out1[0] - out2[0]).abs()) < 1e-6

    def test_positionwise_feedforward_determinism(self):
        pos_ffn = PositionwiseFeedForward(d_model, 2*d_model, dropout=0.0)
        enc_input = torch.ones(10, max_steps, d_model)
        out1 = pos_ffn(enc_input)
        out2 = pos_ffn(enc_input)
        assert torch.max((out1 - out2).abs()) < 1e-6