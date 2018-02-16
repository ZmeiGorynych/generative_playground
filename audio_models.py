import torch
print(torch.__version__)
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
from warpctc_pytorch import CTCLoss

import sys
# that path is the location of the audio generator
sys.path.append("../aind/AIND-VUI-Capstone")

# to_gpu is equivalent to .cuda()
#from gpu_utils import to_gpu
from torch.cuda import FloatTensor, IntTensor

def to_gpu(x):
    return x.cuda()

class LSTMModel(nn.Module):

    def __init__(self, input_dim=None, hidden_dim=200, output_dim=100, batch_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        # The LSTM takes sequences of spectrograms/MFCCs as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = to_gpu(nn.LSTM(input_dim, hidden_dim))

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = to_gpu(nn.Linear(hidden_dim, output_dim))
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
        sz = lstm_out.shape
        # flatten it so
        lin_out = self.hidden2tag(lstm_out.view(-1, sz[2])).view(sz[0],sz[1],-1)

        # tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        # tag_scores = F.log_softmax(tag_space, dim=1)
        return lin_out

if __name__ == '__main__':
    # TODO: do we reset hidden state for each batch?
    from data_generator_short import AudioGenerator2
    spectrogram = False
    batch_size = 20
    output_dim = 29

    # get the data, from a pre-existing generator
    audio_gen = AudioGenerator2(spectrogram=spectrogram,
                                pad_sequences = False,
                                minibatch_size = batch_size)
    audio_gen.load_data(fit_params=True)
    data_gen = audio_gen.gen()
    inputs_, outputs = next(data_gen)

    # reshape the data to our needs
    inputs = torch.FloatTensor(inputs_['the_input']).permute(1,0,2)
    inputs = Variable(to_gpu(inputs))

    probs_sizes = inputs_['input_length'].T[0]
    label_sizes = inputs_['label_length'].T[0]
    labels = []
    for i,row in enumerate(inputs_['the_labels']):
        labels += list(row[:int(label_sizes[i])])

    labels = Variable(IntTensor([int(label) for label in labels]))
    probs_sizes = Variable(IntTensor([int(x) for x in probs_sizes]))
    label_sizes = Variable(IntTensor([int(x) for x in label_sizes]))
    num_features = inputs.shape[2]

    # create and call the model
    model = LSTMModel(num_features,
                      hidden_dim=200,
                      output_dim=output_dim,
                      batch_size = batch_size)
    log_probs = model.forward(inputs)

    ctc_loss = to_gpu(CTCLoss())
    cost = ctc_loss(log_probs, labels, probs_sizes, label_sizes)

    print('aaa')
