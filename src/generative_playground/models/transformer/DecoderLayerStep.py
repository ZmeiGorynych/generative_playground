import torch
from torch import nn as nn
from torch.nn import init as init

from generative_playground.models.transformer.Modules import ScaledDotProductAttention, LayerNormalization, BottleLinear as Linear


class DecoderLayerStep(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, dropout=0.1):
        super().__init__()
        self.enc_attn = MultiHeadAttentionStep(n_head, d_model, d_k, d_v, dropout=dropout)
        self.slf_attn = MultiHeadAttentionStep(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForwardStep(d_model, d_inner_hid, dropout=dropout)

    def forward(self, dec_input, slf_attn_mask=None, dec_enc_attn_mask=None, remember_step=True):
        '''
        One step of a Transformer decoder layer
        :param dec_input: batch x d_model floats
        :param slf_attn_mask:
        :param dec_enc_attn_mask:
        :return: batch x d_model floats
        '''

        dec_output, dec_slf_attn = self.slf_attn(dec_input, attn_mask=slf_attn_mask,remember_step=remember_step)
        if self.encoder_output is not None:
            dec_output, dec_enc_attn = self.enc_attn(dec_output, attn_mask=dec_enc_attn_mask,remember_step=remember_step)
        else:
            dec_enc_attn = None
        # pos_ffn still expects a sequence, though acts on one element at a time, so have to convert
        dec_output = self.pos_ffn(dec_output.unsqueeze(1)).squeeze(1)

        return dec_output, dec_slf_attn, dec_enc_attn

    def init_encoder_output(self, enc_output):
        # must be called at the start of each sequence, so also before any forward() call
        self.encoder_output = enc_output
        self.slf_attn.init_encoder_output(None) # to reset the internal state
        self.enc_attn.init_encoder_output(enc_output)
        # pos_ffn let's leave without caching for now


class MultiHeadAttentionStep(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, enc_output = None):
        '''

        :param n_head:
        :param d_model:
        :param d_k:
        :param d_v:
        :param dropout:
        '''
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Parameter(torch.FloatTensor(n_head, d_model, d_k))
        self.w_ks = nn.Parameter(torch.FloatTensor(n_head, d_model, d_k))
        self.w_vs = nn.Parameter(torch.FloatTensor(n_head, d_model, d_v))

        self.attention = ScaledDotProductAttention()
        self.layer_norm = LayerNormalization(d_model)
        self.proj = Linear(n_head*d_v, d_model)

        self.dropout = nn.Dropout(dropout)

        init.xavier_normal_(self.w_qs)
        init.xavier_normal_(self.w_ks)
        init.xavier_normal_(self.w_vs)

    def init_encoder_output(self, enc_output):
        """
        Pre-calculates k and v for encoder output, to save both time and RAM
        :param enc_output: batch_size x seq_len x d_model
        :return:
        """
        self.dec_q = []
        self.enc_output = enc_output # if that's None, that just resets the state
        if self.enc_output is None:
            self.dec_k = []
            self.dec_v = []
        else:
            self.enc_k = []
            self.enc_v = []
            # pre-calc
            for i in range(self.enc_output.size()[1]):
                _, k, v = self.vec_to_qkv(self.enc_output[:,i,:])  # go over the sequence, collect k and v
                self.enc_k.append(k)
                self.enc_v.append(v)
            self.enc_k = torch.cat(self.enc_k, 1)
            self.enc_v = torch.cat(self.enc_v, 1)

    def vec_to_qkv(self, x):
        '''
        Converts batch of vectors to batches of q,k,v
        :param x: batch_size x d_model
        :return:
        '''
        batch_size = x.size()[0]
        x_nice = x.unsqueeze(0).expand(self.n_head,-1,-1) # result is self.n_head x batch_size x d_model, and no new memory allocation
        q = torch.bmm(x_nice, self.w_qs).transpose(0, 1).reshape(self.n_head*batch_size,1, self.d_k)
        k = torch.bmm(x_nice, self.w_ks).transpose(0, 1).reshape(self.n_head*batch_size,1, self.d_k)
        v = torch.bmm(x_nice, self.w_vs).transpose(0, 1).reshape(self.n_head*batch_size,1, self.d_v)
        return q, k, v


    def forward(self, x, attn_mask=None, remember_step=True):
        '''
        Calculates attention-based output from the next input, caching intermediate results
        :param x: batch_size x d_model
        :param attn_mask:
        :param remember_step: whether to add the last input to input cache
        :return: output same dimension as x
        '''

        self.batch_size = x.size()[0]

        q, k, v = self.vec_to_qkv(x)
        q_s = q #n_head*batch_size x 1 x d_q

        if self.enc_output is None: # this is the self-attention part
            # append the new k, v to the array so far
            self.dec_k.append(k)
            self.dec_v.append(v)
            k_s = torch.cat(self.dec_k,1)  # n_head*batch_size x len_k x d_k
            v_s = torch.cat(self.dec_v,1)  # n_head*batch_size x len_v x d_v
            if not remember_step:
                self.dec_k.pop[-1]
                self.dec_v.pop[-1]
        else: # use the pre-calc'd values
            k_s = self.enc_k
            v_s = self.enc_v

        # perform attention on the latest input, result size# batch_size*n_head x 1 x d_v
        outputs, attns = self.attention(q_s, k_s, v_s, attn_mask=attn_mask.repeat(self.n_head, 1, 1))
        # project back to residual size
        outputs = self.proj(outputs.view(self.batch_size, -1)) # batch_size x n_head*d_v
        outputs = self.dropout(outputs)

        return self.layer_norm(outputs + x), attns


class PositionwiseFeedForwardStep(nn.Module):
    """
    A two-feed-forward-layer module

    """

    def __init__(self, d_model, d_inner_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_inner_hid)
        self.w_2 = nn.Linear(d_inner_hid, d_model)
        self.layer_norm = LayerNormalization(d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward step
        :param x: batch_size x d_model
        :return: batch_size x d_model
        """
        # x is batch of vectors, so need no dimension juggling
        output = self.relu(self.w_1(x))
        output = self.w_2(output)
        output = self.dropout(output)
        return self.layer_norm(output + x)