import torch
from torch import nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

print(torch.__version__)
import torch.nn as nn
import torch.autograd as autograd

from basic_pytorch.gpu_utils import to_gpu, FloatTensor#, LongTensor

class LSTMModel(nn.Module):
    def __init__(self,
                 input_dim=None,
                 hidden_dim=200,
                 output_dim=100,
                 batch_size=1,
                 p_dropout = 0.2,
                 num_layers = 2):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.bidirectional = True
        self.num_layers = num_layers
        self.bidir_mult = 2 if self.bidirectional else 1
        self.dimension_mult = self.num_layers * self.bidir_mult
        # The LSTM takes sequences of spectrograms/MFCCs as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = to_gpu(nn.LSTM(input_dim, hidden_dim,
                                   bidirectional=self.bidirectional,
                                   num_layers=self.num_layers,
                                   dropout=p_dropout))
        self.dropout_1 = nn.Dropout(p_dropout)
        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = to_gpu(nn.Linear(hidden_dim*self.bidir_mult, output_dim))
        self.reset_hidden()

    def init_hidden(self, batch_size = None):
        if batch_size is None:
            batch_size = self.batch_size
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(to_gpu(torch.zeros(self.dimension_mult, batch_size, self.hidden_dim))),
                autograd.Variable(to_gpu(torch.zeros(self.dimension_mult, batch_size, self.hidden_dim)))
                )

    def reset_hidden(self):
        self.hidden = self.init_hidden()

    def forward(self, ext_inputs):
        inputs, input_sizes = ext_inputs
        lstm_out, self.hidden = self.lstm(inputs, self.hidden)
        sz = lstm_out.shape
        # flatten it so
        lstm_out = self.dropout_1(lstm_out)
        lin_out = self.hidden2tag(lstm_out.view(-1, sz[2])).view(sz[0],sz[1],-1)
        prob_out = lin_out#torch.nn.functional.log_softmax(lin_out, dim=2)
        # tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        # tag_scores = F.log_softmax(tag_space, dim=1)
        return prob_out, input_sizes


class SimpleRNNDecoder(nn.Module):
    # implementation matches model_eq.py _buildDecoder, at least in intent
    def __init__(self,
                 z_size=200,
                 hidden_n=200,
                 feature_len=12,
                 max_seq_length=15,
                 drop_rate = 0.0,
                 num_layers = 3):
        super(SimpleRNNDecoder, self).__init__()
        self.max_seq_length = max_seq_length
        self.hidden_n = hidden_n
        self.num_layers = num_layers
        self.output_feature_size = feature_len
        # TODO: is the batchNorm applied on the correct dimension?
        self.batch_norm = nn.BatchNorm1d(z_size)
        self.fc_input = nn.Linear(z_size, hidden_n)
        self.dropout_1 = nn.Dropout(drop_rate)
        self.gru_1 = nn.GRU(input_size=hidden_n,
                            hidden_size=hidden_n,
                            batch_first=True,
                            dropout=drop_rate,
                            num_layers=self.num_layers)
        self.dropout_2 = nn.Dropout(drop_rate)
        self.fc_out = nn.Linear(hidden_n, feature_len)
        self.hidden = None

    def forward(self, encoded):#, hidden_1 = None):
        batch_size = len(encoded)
        if self.hidden is None:
            self.hidden = self.init_hidden(batch_size=batch_size)

        # copy the latent state to length of sequence, instead of sampling inputs
        embedded = F.relu(self.fc_input(self.batch_norm(encoded))) \
            .view(batch_size, 1, self.hidden_n) \
            .repeat(1, self.max_seq_length, 1)
        embedded =self.dropout_1(embedded)
        # run the GRU on it
        out_3, self.hidden = self.gru_1(embedded, self.hidden)
        # tmp has dim (batch_size*seq_len)xhidden_n, so we can apply the linear transform to it
        tmp = self.dropout_2(out_3.contiguous().view(-1, self.hidden_n))
        out = self.fc_out(tmp).view(batch_size,
                                    self.max_seq_length,
                                    self.output_feature_size)

        # just return the logits
        #self.hidden = None
        return out#, hidden_1

    def reset_state(self):
        self.hidden = None

    # TODO: remove this method!
    def decode(self, z):
        if 'numpy' in str(type(z)):
            z = Variable(FloatTensor(z))
        self.reset_state()
        output = self.forward(z)
        return output.data.cpu().numpy()

    def init_hidden(self, batch_size):
        # NOTE: assume only 1 layer no bi-direction
        h1 = Variable(to_gpu(torch.zeros(self.num_layers, batch_size, self.hidden_n)), requires_grad=False)
        return h1

class ResettingRNNDecoder(SimpleRNNDecoder):
    def forward(self, encoded):
        out = super().forward(encoded)
        self.reset_state()
        return out

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
        self.num_layers = num_layers
        self.feature_size = feature_len
        self.bidirectional = bidirectional
        self.z_size = z_size
        self.bidirectional = bidirectional
        self.bidir_mult = 2 if self.bidirectional else 1
        self.dimension_mult = self.num_layers * self.bidir_mult

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
        self.fc_out = nn.Linear(hidden_n*self.bidir_mult, z_size)

    def forward(self, input_seq, hidden = None):
        # input_seq: batch_size x seq_len x feature_len
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