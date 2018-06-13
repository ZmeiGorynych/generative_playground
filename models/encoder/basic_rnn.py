import torch
from torch import nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

from generative_playground.gpu_utils import to_gpu


class SimpleRNNAttentionEncoder(nn.Module):
    # implementation matches model_eq.py _buildDecoder, at least in intent
    def __init__(self,
                 z_size=200,
                 hidden_n=200,
                 feature_len=12,
                 max_seq_length=15,
                 drop_rate = 0.0,
                 num_layers = 3,
                 bidirectional = True):
        super(SimpleRNNAttentionEncoder, self).__init__()
        self.max_seq_length = max_seq_length
        self.hidden_n = hidden_n
        self.z_size = z_size
        self.num_layers = num_layers
        self.feature_size = feature_len
        self.bidirectional = bidirectional
        self.z_size = z_size
        self.bidirectional = bidirectional
        self.bidir_mult = 2 if self.bidirectional else 1
        self.dimension_mult = self.num_layers * self.bidir_mult
        self.output_shape = [None, hidden_n]

        # TODO: is the batchNorm applied on the correct dimension?
        # self.batch_norm = nn.BatchNorm1d(z_size)
        # self.fc_input = nn.Linear(z_size, hidden_n)
        # self.dropout_1 = nn.Dropout(drop_rate)
        self.gru_1 = nn.GRU(input_size=feature_len,
                            hidden_size=hidden_n,
                            batch_first=True,
                            dropout=drop_rate,
                            num_layers=self.num_layers,
                            bidirectional=bidirectional)
        self.dropout_1 = nn.Dropout(drop_rate)
        self.dropout_2 = nn.Dropout(drop_rate)
        #self.dropout_3 = nn.Dropout(drop_rate)
        self.attention_wgt = nn.Linear(hidden_n*self.bidir_mult,1)
        self.fc_out = nn.Linear(hidden_n*self.bidir_mult, hidden_n)

    def forward(self, input_seq, hidden=None):
        '''

        :param input_seq: batch_size x seq_len x feature_len
        :param hidden: hidden state from previous state - do we ever use that?
        :return: batch_size x hidden_n
        '''
        # input_seq:
        batch_size = input_seq.size()[0]
        if hidden is None:
            hidden = self.init_hidden(batch_size)
        # copy the latent state to length of sequence, instead of sampling inputs

        # run the GRU on it
        gru_out, hidden = self.gru_1(input_seq, hidden)
        gru_out = self.dropout_1(gru_out.contiguous())
        pre_attn_wgt = self.attention_wgt(gru_out.view(-1,self.hidden_n*self.bidir_mult))
        attn_wgt = F.softmax(pre_attn_wgt.view(batch_size,self.max_seq_length,1), dim=1)
        # transform each hidden state to output dimension
        attn_inputs = self.fc_out(gru_out.view(-1,self.hidden_n*self.bidir_mult))\
                                .view(batch_size,self.max_seq_length,self.z_size)
        out = (attn_wgt*self.dropout_2(attn_inputs)).sum(1)
        # just return the logits
        return out, hidden

    def encode(self,x):
        '''

        :param x: a numpy array batch x seq x feature
        :return:
        '''
        out, hidden = self.forward(to_gpu(Variable(FloatTensor(x))))
        return out.data.cpu().numpy()

    def init_hidden(self, batch_size):
        h1 = Variable(to_gpu(torch.zeros(self.dimension_mult, batch_size, self.hidden_n)), requires_grad=False)
        return h1