from torch import nn as nn
from torch.nn import functional as F

from generative_playground.utils.gpu_utils import to_gpu


class MeanVarianceSkewHead(nn.Module):
    def __init__(self, model, output_dim, drop_rate=0.2):
        super().__init__()
        self.z_size = output_dim
        self.model = model
        #self.dropout = nn.Dropout(drop_rate)
        self.fc_mu = to_gpu(nn.Linear(self.model.output_shape[-1], output_dim))
        self.fc_var = to_gpu(nn.Linear(self.model.output_shape[-1], output_dim))
        self.fc_skew = to_gpu(nn.Linear(self.model.output_shape[-1], output_dim))
        self.output_shape = [None, output_dim]

    def forward(self, x):
        '''

        :param x: either batch x num_dims or batch x 1 x num_dims
        :return: two vectors each of size batch x output_dim
        '''
        out = F.relu(self.model.forward(x))
        # TODO: remove the necessity for the below!
        # TODO: handle sequences!
        if isinstance(out,tuple) or isinstance(out, list):
            out = out[0]

        #out = self.dropout(out)
        my_size = out.size()
        if len(my_size)==3 and my_size[1]==1:
            #flatten sequences of one element
            out = out.view(-1, out.size[-1])

        mu = self.fc_mu(out)
        log_var = self.fc_var(out)
        skew = self.fc_skew(out)
        return mu, log_var, skew