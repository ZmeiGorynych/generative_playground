import torch
from torch import nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

from generative_playground.data_utils.to_one_hot import to_one_hot
from generative_playground.gpu_utils import to_gpu, FloatTensor


class SimpleRNNDecoder(nn.Module):
    # implementation matches model_eq.py _buildDecoder, at least in intent
    def __init__(self,
                 z_size=200,
                 hidden_n=200,
                 feature_len=12,
                 max_seq_length=15,
                 drop_rate = 0.0,
                 num_layers = 3,
                 use_last_action = False):
        super(SimpleRNNDecoder, self).__init__()
        self.max_seq_length = max_seq_length
        self.z_size = z_size
        self.hidden_n = hidden_n
        self.num_layers = num_layers
        self.output_feature_size = feature_len
        self.use_last_action = use_last_action
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

    def encode(self, enc_output, last_action):
        if not self.use_last_action:
            return enc_output
        else:
            if last_action is not None and last_action[0] is not None:
                # if the above is false, it uses the original value of self.one_hot_action, which is zeros
                self.one_hot_action = to_one_hot(last_action,
                                            n_dims=self.output_feature_size,
                                            out=self.one_hot_action)

            encoded = torch.cat([enc_output, self.one_hot_action], 1)

        return encoded


    def forward(self, last_action=None, last_action_pos=None):
        '''
        One step of the RNN model
        :param enc_output: batch x z_size, so don't support sequences
        :param last_action: batch of ints, all equaling None for first step
        :param last_action_pos: ignored, used by the attention decoder, here just to get the signature right
        :return:
        '''
        # check we don't exceed max sequence length
        if self.n == self.max_seq_len:
            raise StopIteration()
        self.n+=1

        if self.hidden is None: # first step after reset
            # need to do it here as batch size might be different for each sequence
            self.hidden = self.init_hidden(batch_size=self.batch_size)
            self.one_hot_action = to_gpu(torch.zeros(self.batch_size, self.output_feature_size))

        encoded = self.encode(self.enc_output, last_action)

        # copy the latent state to length of sequence, instead of sampling inputs
        embedded = F.relu(self.fc_input(self.batch_norm(encoded))) \
            .view(self.batch_size, 1, self.hidden_n) \
            .repeat(1, self.max_seq_length, 1)
        embedded =self.dropout_1(embedded)
        # run the GRU on it
        out_3, self.hidden = self.gru_1(embedded, self.hidden)
        # tmp has dim (batch_size*seq_len)xhidden_n, so we can apply the linear transform to it
        tmp = self.dropout_2(out_3.contiguous().view(-1, self.hidden_n))
        out = self.fc_out(tmp).view(self.batch_size,
                                    self.max_seq_length,
                                    self.output_feature_size)

        # just return the logits
        #self.hidden = None
        return out#, hidden_1

    def init_encoder_output(self,z):
        '''
        Must be called at the start of each new sequence
        :param z:
        :return:
        '''
        self.hidden = None
        self.enc_output = z
        self.batch_size = z.size()[0]
        self.z_size = z.size()[-1]
        self.n = 0

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