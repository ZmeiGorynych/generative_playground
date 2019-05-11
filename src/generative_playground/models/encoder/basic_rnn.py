import torch
from torch import nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from generative_playground.models.heads.attention_aggregating_head import AttentionAggregatingHead
from generative_playground.utils.gpu_utils import to_gpu

class SimpleRNN(nn.Module):
    def __init__(self,
                 hidden_n=200,
                 feature_len=12,
                 drop_rate=0.0,
                 num_layers=3,
                 bidirectional=True):
        super().__init__()
        self.hidden_n = hidden_n
        bidir_mult = 2 if bidirectional else 1
        self.normalize_output = nn.Linear(hidden_n * bidir_mult, hidden_n)
        self.dropout = nn.Dropout(drop_rate)
        self.dimension_mult = num_layers * bidir_mult
        self.output_shape = [None, None, hidden_n]

        # TODO: is the batchNorm applied on the correct dimension?
        # self.batch_norm = nn.BatchNorm1d(z_size)
        self.gru_1 = nn.GRU(input_size=feature_len,
                            hidden_size=hidden_n,
                            batch_first=True,
                            dropout=drop_rate,
                            num_layers=num_layers,
                            bidirectional=bidirectional)

        # https: // github.com / pytorch / pytorch / issues / 9221
        def init_weights(m):
            if hasattr(m, 'weight'):
                torch.nn.init.xavier_uniform(m.weight)

        self.gru_1.apply(init_weights)

    def forward(self, input_seq, hidden=None):
        '''

        :param input_seq: batch_size x seq_len x feature_len
        :param hidden: hidden state from earlier
        :return: batch_size x hidden_n
        '''
        batch_size = input_seq.size()[0]
        if hidden is None:
            hidden = Variable(to_gpu(torch.zeros(self.dimension_mult,
                                                 batch_size,
                                                 self.hidden_n)),
                              requires_grad=False)
            # self.init_hidden(batch_size)

        # run the GRU on it
        gru_out, hidden = self.gru_1(input_seq, hidden)
        out = self.normalize_output(self.dropout(F.relu(gru_out)))
        return out

    # def init_hidden(self, batch_size):
    #     h1 = Variable(to_gpu(torch.zeros(self.dimension_mult, batch_size, self.hidden_n)), requires_grad=False)
    #     return h1
