import torch
import torch.nn as nn
import torch.autograd as autograd

import sys
sys.path.append("../aind/AIND-VUI-Capstone")

from gpu_utils import to_gpu

class LSTMModel(nn.Module):

    def __init__(self, input_dim=None, hidden_dim=200, output_dim=200, batch_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        # The LSTM takes sequences of spectrograms/MFCCs as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(input_dim, hidden_dim).cuda()

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, output_dim).cuda()
        self.hidden = self.init_hidden()

    def init_hidden(self, batch_size = None):
        if batch_size is None:
            batch_size = self.batch_size
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(to_gpu(torch.zeros(1, batch_size, self.hidden_dim))),
                autograd.Variable(to_gpu(torch.zeros(1, batch_size, self.hidden_dim)))
                )

    def forward(self, inputs):
        lstm_out, self.hidden = self.lstm(inputs, self.hidden)
           # embeds.view(len(sentence), 1, -1), self.hidden)

        # tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        # tag_scores = F.log_softmax(tag_space, dim=1)
        return lstm_out

if __name__ == '__main__':
    # TODO: do we reset hidden state for each batch?
    from data_generator_short import AudioGenerator2
    spectrogram = False
    batch_size = 20
    audio_gen = AudioGenerator2(spectrogram=spectrogram,
                                pad_sequences = False,
                                minibatch_size = batch_size)
    audio_gen.load_data(fit_params=True)
    data_gen = audio_gen.gen()
    inputs_, outputs = next(data_gen)
    inputs = autograd.Variable(to_gpu(torch.FloatTensor(inputs_['the_input']).permute(1,0,2)))
    num_features = inputs.shape[2]
    model = LSTMModel(num_features,200,200, batch_size = batch_size)
    x = model.forward(inputs)
    print('aaa')
