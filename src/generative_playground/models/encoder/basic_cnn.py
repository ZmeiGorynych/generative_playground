from torch import nn as nn
from torch.nn import functional as F


class SimpleCNNEncoder(nn.Module):
    def __init__(self,
                 max_seq_length=None,
                 params = {},
                 feature_len=None,
                 drop_rate = 0.0):
        super(SimpleCNNEncoder, self).__init__()
        self.k = params['kernel_sizes']
        self.ch = params['filters']
        self.dense_size = params['dense_size']
        # NOTE: GVAE implementation does not use max-pooling. Original DCNN implementation uses max-k pooling.
        #conv_args = {'in_channels':feature_len, 'out_channels':feature_len,  'groups':feature_len}
        self.dropout1 = nn.Dropout(drop_rate)
        self.conv_1 = nn.Conv1d(kernel_size=self.k[0],
                                in_channels=feature_len,
                                out_channels=self.ch[0])
        self.bn_1 = nn.BatchNorm1d(self.ch[0])
        self.dropout2 = nn.Dropout(drop_rate)

        self.conv_2 = nn.Conv1d(kernel_size=self.k[1],
                                in_channels=self.ch[0],
                                out_channels=self.ch[1])
        self.bn_2 = nn.BatchNorm1d(self.ch[1])
        self.dropout3 = nn.Dropout(drop_rate)
        self.conv_3 = nn.Conv1d(kernel_size=self.k[2],
                                in_channels=self.ch[1],
                                out_channels=self.ch[2])
        self.bn_3 = nn.BatchNorm1d(self.ch[2])
        self.dropout4 = nn.Dropout(drop_rate)

        self.fc_0 = nn.Linear(self.ch[2] * (max_seq_length + len(self.k) - sum(self.k)), self.dense_size)
        self.dropout5 = nn.Dropout(drop_rate)
        self.output_shape = [None, self.dense_size]

    def forward(self, x):
        '''

        :param x: Sequence to encode, batch_size x seq_len x d_input floats or batch_size x seq_len ints
        :return:
        '''
        batch_size = x.size()[0]
        # Conv1D expects dimension batch x channels x feature
        # we treat the one-hot encoding as channels, but only convolve one channel at a time?
        # why not just view() the array into the right shape?
        x = x.transpose(1, 2)#.contiguous()
        x = self.dropout1(x)
        x = F.relu(self.bn_1(self.conv_1(x)))
        x = self.dropout2(x)
        x = F.relu(self.bn_2(self.conv_2(x)))
        x = self.dropout3(x)
        x = F.relu(self.bn_3(self.conv_3(x)))
        x = self.dropout4(x)
        x_ = x.view(batch_size, -1)
        h = F.relu(self.fc_0(x_))
        h = self.dropout5(h)
        return h

    # def encode(self,x):
    #     '''
    #     # TODO this is broken for now, as mu and sigma are now mapped inside vae
    #     :param x: a numpy array batch x seq x feature
    #     :return:
    #     '''
    #     mu_, var_ = self.forward(Variable(FloatTensor(x)))
    #     return mu_.data.cpu().numpy()