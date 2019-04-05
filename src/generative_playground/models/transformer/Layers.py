''' Define the Layers '''
import torch.nn as nn
import torch
from generative_playground.models.transformer.SubLayers import MultiHeadAttention, \
    PositionwiseFeedForward

__author__ = "Yu-Hsiang Huang"

class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, dropout=0.1, n_max_seq=None,
                 use_attentions=None):
        super(EncoderLayer, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.use_attentions = use_attentions
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner_hid, dropout=dropout)
        if self.use_attentions is not None and self.use_attentions:
            self.n_max_seq = n_max_seq
            both_mult = 2 if self.use_attentions=='both' else 1
            self.attn_fc = nn.Linear(d_model + both_mult*n_head*n_max_seq, d_model)

    def forward(self, enc_input, slf_attn_mask=None):
        batch_size = enc_input.size()[0]
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, attn_mask=slf_attn_mask)
        if self.use_attentions is not None and self.use_attentions:
            self_attn_nice = reshape_self_attention(enc_slf_attn,
                                                    self.n_head,
                                                    batch_size,
                                                    self.n_max_seq,
                                                    self.use_attentions)
            enc_output = torch.cat([self_attn_nice.detach(), enc_output], dim=2)
            enc_output = self.attn_fc(enc_output)
        enc_output = self.pos_ffn(enc_output)

        return enc_output, enc_slf_attn

def reshape_self_attention(x, n_head, batch_size, n_max_seq, transpose):
    x1 = x.view(n_head, batch_size, n_max_seq, n_max_seq)
    if transpose is True:
        x1 = x1.transpose(2,3)
    elif transpose == 'both':
        x1 = torch.cat([x1, x1.transpose(2, 3)], dim=3)

    x2 = x1.transpose(0,1).transpose(1,2).contiguous().view(batch_size, n_max_seq, -1)
    return x2

class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner_hid, dropout=dropout)

    def forward(self, dec_input, enc_output, slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, attn_mask=slf_attn_mask)
        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, attn_mask=dec_enc_attn_mask)
        dec_output = self.pos_ffn(dec_output)

        return dec_output, dec_slf_attn, dec_enc_attn


