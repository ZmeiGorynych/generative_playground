''' Define the Transformer model '''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from torch._C import device
from generative_playground.utils.gpu_utils import device
import generative_playground.models.transformer.Constants as Constants
from generative_playground.models.embedder.embedder import Embedder
from generative_playground.models.transformer.Layers import EncoderLayer, DecoderLayer

__author__ = "Yu-Hsiang Huang, much refactored and amended by Egor Kraev"


def get_attn_padding_mask(seq_q, seq_k, padding_idx=Constants.PAD):
    ''' Indicate the padding-related part to mask '''
    assert seq_q.dim() == 2 and seq_k.dim() == 2
    mb_size, len_q = seq_q.size()
    mb_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(padding_idx).unsqueeze(1)   # bx1xsk
    pad_attn_mask = pad_attn_mask.expand(mb_size, len_q, len_k) # bxsqxsk
    return pad_attn_mask

def get_attn_subsequent_mask(seq):
    ''' Get an attention mask to avoid using the subsequent info.'''
    assert seq.dim() == 2
    attn_shape = (seq.size(0), seq.size(1), seq.size(1))
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    subsequent_mask = torch.from_numpy(subsequent_mask)
    if seq.is_cuda:
        subsequent_mask = subsequent_mask.cuda()
    return subsequent_mask

def get_attn_padding_seq_from_tensor(x, padding_idx=Constants.PAD):
    '''
    Takes a 3D tensor and creates a mask, treating all-zero features as padding
    :param x:
    :param padding_idx:
    :return:
    '''
    assert len(x.size()) == 3
    non_pad_idx = 1 if padding_idx == 0 else 0
    out = torch.ones(*(x.size()[:2]), dtype=torch.long, device=device)*non_pad_idx
    elems_for_masking = (torch.sum(torch.abs(x),dim=2) == 0)
    out[elems_for_masking] = padding_idx
    return out

class TransformerEncoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(self,
                 n_src_vocab=None,  # feature_len
                 n_max_seq=None, # only required when use_self_attention = True # TODO: is it needed even then?
                 n_layers=6,  #6,
                 n_head=8,  #6,
                 d_k=64,  #16,
                 d_v=64,  #16,#
                 d_model=512,  #128,#
                 dropout=0.1,
                 use_self_attention=False,
                 transpose_self_attention=False,
                 embedder=None,
                 padding_idx=Constants.PAD # TODO: remember to set this to n_src_vocab-1 when calling from my code!
                 ):

        super(TransformerEncoder, self).__init__()
        if embedder is not None:
            self.embedder=embedder
        else:
            self.embedder = Embedder(n_max_seq,
                                 n_src_vocab,
                                 d_model,
                                 padding_idx)
        d_inner_hid = 2*d_model
        self.n_max_seq = n_max_seq
        self.d_model = d_model
        self.n_head = n_head
        self.include_self_attention = use_self_attention
        self.transpose_self_attention = transpose_self_attention
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model,
                         d_inner_hid,
                         n_head,
                         d_k,
                         d_v,
                         dropout=dropout,
                         n_max_seq=n_max_seq,
                         use_attentions=use_self_attention)
            for _ in range(n_layers)])
        # both_mult = 2 if transpose_self_attention=='both' else 1
        # self.final_fc = nn.Linear(d_model + both_mult*n_head*n_max_seq*n_layers, d_model)

        self.output_shape = [None, n_max_seq, d_model]
        self.to(device=device)

    def forward(self, *args, **kwargs):
        '''

        :param src_seq: batch_size x seq_len x feature_len float (eg one-hot) or batch_size x seq_len ints
        :param src_pos: batch_size x seq_len ints, optional
        :param return_attns:
        :return: batch_size x n_max_seq x d_model
        '''

        enc_input = self.embedder(*args, **kwargs)
        if isinstance(enc_input, tuple):
            enc_input, src_seq_for_masking = enc_input
        else:
            src_seq_for_masking = get_attn_padding_seq_from_tensor(enc_input)
        # if self.include_self_attention:
        #     enc_slf_attns = []

        enc_output = enc_input
        enc_slf_attn_mask = get_attn_padding_mask(src_seq_for_masking, src_seq_for_masking)
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output, slf_attn_mask=enc_slf_attn_mask)
            # if self.include_self_attention:
            #     enc_slf_attns += [enc_slf_attn]

        # if self.z_size is not None:
        #     enc_output = self.z_enc(enc_output.view(batch_size*seq_len,-1)).view(batch_size,seq_len,-1)

        # if False:#self.include_self_attention:
        #     nice_attentions = [reshape_self_attention(x,
        #                                               self.n_head,
        #                                               len(enc_output),
        #                                               self.n_max_seq,
        #                                               self.transpose_self_attention) for x in enc_slf_attns]
        #     enc_output = torch.cat([enc_output] + nice_attentions, dim=2)
        #     enc_output = self.final_fc(F.relu(enc_output))

        return enc_output

def reshape_self_attention(x, n_head, batch_size, n_max_seq, transpose):
    x1 = x.view(n_head, batch_size, n_max_seq, n_max_seq)
    if transpose is True:
        x1 = x1.transpose(2,3)
    elif transpose == 'both':
        x1 = torch.cat([x1, x1.transpose(2, 3)], dim=3)

    x2 = x1.transpose(0,1).transpose(1,2).contiguous().view(batch_size, n_max_seq, -1)
    return x2

class TransformerDecoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''
    def __init__(
            self, n_tgt_vocab, n_max_seq, n_layers=6, n_head=8, d_k=64, d_v=64,
            d_word_vec=512, d_model=512, d_inner_hid=1024, dropout=0.1):

        super(TransformerDecoder, self).__init__()
        n_position = n_max_seq + 1
        self.n_max_seq = n_max_seq
        self.d_model = d_model

        self.position_enc = nn.Embedding(
            n_position, d_word_vec, padding_idx=Constants.PAD)
        self.position_enc.weight.data = position_encoding_init(n_position, d_word_vec)

        self.tgt_word_emb = nn.Embedding(
            n_tgt_vocab, d_word_vec, padding_idx=Constants.PAD)
        self.dropout = nn.Dropout(dropout)

        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, tgt_seq, tgt_pos, src_seq, enc_output, return_attns=False):
        # Word embedding look up
        dec_input = self.tgt_word_emb(tgt_seq)

        # Position Encoding addition
        dec_input += self.position_enc(tgt_pos)

        # Decode
        dec_slf_attn_pad_mask = get_attn_padding_mask(tgt_seq, tgt_seq)
        dec_slf_attn_sub_mask = get_attn_subsequent_mask(tgt_seq)
        dec_slf_attn_mask = torch.gt(dec_slf_attn_pad_mask + dec_slf_attn_sub_mask, 0)

        dec_enc_attn_pad_mask = get_attn_padding_mask(tgt_seq, src_seq)

        if return_attns:
            dec_slf_attns, dec_enc_attns = [], []

        dec_output = dec_input
        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output,
                slf_attn_mask=dec_slf_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_pad_mask)

            if return_attns:
                dec_slf_attns += [dec_slf_attn]
                dec_enc_attns += [dec_enc_attn]

        if return_attns:
            return dec_output, dec_slf_attns, dec_enc_attns
        else:
            return dec_output,


# class Transformer(nn.Module):
#     ''' A sequence to sequence model with attention mechanism. '''
#
#     def __init__(
#             self, n_src_vocab, n_tgt_vocab, n_max_seq, n_layers=6, n_head=8,
#             d_word_vec=512, d_model=512, d_inner_hid=1024, d_k=64, d_v=64,
#             dropout=0.1, proj_share_weight=True, embs_share_weight=True):
#
#         super(Transformer, self).__init__()
#         self.encoder = TransformerEncoder(
#             n_src_vocab, n_max_seq, n_layers=n_layers, n_head=n_head,
#             d_word_vec=d_word_vec, d_model=d_model,
#             d_inner_hid=d_inner_hid, dropout=dropout)
#         self.decoder = TransformerDecoder(
#             n_tgt_vocab, n_max_seq, n_layers=n_layers, n_head=n_head,
#             d_word_vec=d_word_vec, d_model=d_model,
#             d_inner_hid=d_inner_hid, dropout=dropout)
#         self.tgt_word_proj = Linear(d_model, n_tgt_vocab, bias=False)
#         self.dropout = nn.Dropout(dropout)
#
#         assert d_model == d_word_vec, \
#         'To facilitate the residual connections, \
#          the dimensions of all module output shall be the same.'
#
#         if proj_share_weight:
#             # Share the weight matrix between tgt word embedding/projection
#             assert d_model == d_word_vec
#             self.tgt_word_proj.weight = self.decoder.tgt_word_emb.weight
#
#         if embs_share_weight:
#             # Share the weight matrix between src/tgt word embeddings
#             # assume the src/tgt word vec size are the same
#             assert n_src_vocab == n_tgt_vocab, \
#             "To share word embedding table, the vocabulary size of src/tgt shall be the same."
#             self.encoder.src_word_emb.weight = self.decoder.tgt_word_emb.weight
#
#     #TODO: just set requires_grad to False instead
#     def get_trainable_parameters(self):
#         ''' Avoid updating the position encoding '''
#         enc_freezed_param_ids = set(map(id, self.encoder.position_enc.parameters()))
#         dec_freezed_param_ids = set(map(id, self.decoder.position_enc.parameters()))
#         freezed_param_ids = enc_freezed_param_ids | dec_freezed_param_ids
#         return (p for p in self.parameters() if id(p) not in freezed_param_ids)
#
#     def forward(self, src, tgt):
#         src_seq, src_pos = src
#         tgt_seq, tgt_pos = tgt
#
#         tgt_seq = tgt_seq[:, :-1]
#         tgt_pos = tgt_pos[:, :-1]
#
#         enc_output, *_ = self.encoder(src_seq, src_pos)
#         dec_output, *_ = self.decoder(tgt_seq, tgt_pos, src_seq, enc_output)
#         seq_logit = self.tgt_word_proj(dec_output)
#
#         return seq_logit.view(-1, seq_logit.size(2))
