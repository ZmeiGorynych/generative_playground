import torch
from torch import nn as nn
from torch.autograd import Variable
from torch.nn import LayerNorm, functional as F

from generative_playground.data_utils.to_one_hot import to_one_hot
from generative_playground.utils.gpu_utils import to_gpu


class NormGRUStepLayer(nn.Module):
    def __init__(self,
                 hidden_n = 200,
                 drop_rate = 0.1):
        super().__init__()
        self.hidden_n = hidden_n
        self.gru = nn.GRU(input_size=hidden_n,
                            hidden_size=hidden_n,
                            batch_first=True,
                            num_layers=1)
        self.output_shape = [None, 1, hidden_n]
        self.layer_norm = LayerNorm(self.output_shape[1:])
        self.dropout = nn.Dropout(drop_rate)
        self.hidden = None

    def forward(self, x, remember_step=True):
        out_1, new_hidden = self.gru(x, self.hidden)
        if remember_step:
            self.hidden = new_hidden
        out_2 = self.dropout(out_1)
        out_3 = self.layer_norm(out_2 + x)
        return out_3

    def reset_state(self, batch_size):
        self.hidden = self.init_hidden(batch_size)

    def init_hidden(self, batch_size):
        # NOTE: assume only 1 layer no bi-direction
        h1 = Variable(to_gpu(torch.zeros(1, batch_size, self.hidden_n)), requires_grad=False)
        return h1


class ResNetRNNDecoder(nn.Module):
    # implementation matches model_eq.py _buildDecoder, at least in intent
    def __init__(self,
                 z_size=200,
                 hidden_n=200,
                 feature_len=12,
                 max_seq_length=15,  # total max sequence length
                 steps=1,  # how many steps to do at each call
                 drop_rate=0.0,
                 num_layers=3,
                 use_last_action=False):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.steps = steps
        if use_last_action:
            eff_z_size = z_size + feature_len
        else:
            eff_z_size = z_size
        self.z_size = z_size
        self.hidden_n = hidden_n
        self.num_layers = num_layers
        #self.output_feature_size = feature_len
        self.use_last_action = use_last_action
        # TODO: is the batchNorm applied on the correct dimension?
        #self.batch_norm = nn.BatchNorm1d(eff_z_size)
        self.fc_input = nn.Linear(eff_z_size, hidden_n)
        self.dropout_1 = nn.Dropout(drop_rate)
        self.layer_stack = nn.ModuleList([NormGRUStepLayer(hidden_n, drop_rate) for _ in range(num_layers)])
        #self.fc_out = nn.Linear(hidden_n, feature_len)
        self.output_shape = [None, 1, hidden_n] #[None, 1, feature_len]

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

    def forward(self, last_action=None, last_action_pos=None, remember_step=True):
        '''
        One step of the RNN model
        :param enc_output: batch x z_size, so don't support sequences
        :param last_action: batch of ints, all equaling None for first step
        :param last_action_pos: ignored, used by the attention decoder, here just to get the signature right
        :return: batch x steps x feature_len
        '''
        # check we don't exceed max sequence length
        if self.n == self.max_seq_length:
            raise StopIteration()
        if remember_step:
            self.n += self.steps

        if self.one_hot_action is None:  # first step after reset
            # need to do it here as batch size might be different for each sequence
            self.one_hot_action = to_gpu(torch.zeros(self.batch_size, self.output_feature_size))

        encoded = self.encode(self.enc_output, last_action)

        # copy the latent state to length of sequence, instead of sampling inputs
        embedded = F.relu(self.fc_input(
                                        #self.batch_norm(encoded)
                                        encoded
                                        )) \
            .view(self.batch_size, 1, self.hidden_n)# \
            #.repeat(1, self.steps, 1)
        out = self.dropout_1(embedded)
        # run the GRUs on it
        for dec_layer in self.layer_stack:
            out = dec_layer(out,remember_step)
        # tmp has dim (batch_size*seq_len)xhidden_n, so we can apply the linear transform to it
        #tmp = self.dropout_2(out.contiguous().view(-1, self.hidden_n))

        # Don't need the below as the vae head maps to output_feature_size
        # tmp = out.contiguous().view(-1, self.hidden_n)
        # out = self.fc_out(tmp).view(self.batch_size,
        #                             1,
        #                             self.output_feature_size)


        return out

    def init_encoder_output(self, z):
        '''
        Must be called at the start of each new sequence
        :param z:
        :return:
        '''
        self.one_hot_action = None
        self.enc_output = z
        self.batch_size = z.size()[0]
        for dec_layer in self.layer_stack:
            dec_layer.reset_state(self.batch_size)
        self.z_size = z.size()[-1]
        self.n = 0