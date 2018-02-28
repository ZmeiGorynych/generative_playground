import torch
print(torch.__version__)
import torch.nn as nn
import torch.autograd as autograd

from gpu_utils import to_gpu

class LSTMModel(nn.Module):
    def __init__(self, input_dim=None, hidden_dim=200, output_dim=100, batch_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.bidirectional = True
        self.num_layers = 1
        self.bidir_mult = 2 if self.bidirectional else 1
        self.dimension_mult = self.num_layers * self.bidir_mult
        # The LSTM takes sequences of spectrograms/MFCCs as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = to_gpu(nn.LSTM(input_dim, hidden_dim,
                                   bidirectional=self.bidirectional,
                                   num_layers=self.num_layers))

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
        lin_out = self.hidden2tag(lstm_out.view(-1, sz[2])).view(sz[0],sz[1],-1)
        prob_out = lin_out#torch.nn.functional.log_softmax(lin_out, dim=2)
        # tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        # tag_scores = F.log_softmax(tag_space, dim=1)
        return prob_out, input_sizes

