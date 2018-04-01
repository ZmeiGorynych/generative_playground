import math
import torch.nn.functional as F
import torch

# multiplies all inputs by a fixed vector
from torch import nn as nn


class Net(torch.nn.Module):
    def __init__(self, w):
        super().__init__()
        self.w = torch.nn.Parameter(w)

    def forward(self, x):
        if len(x.size())==2: # if got a single matrix
            return x @ self.w
        elif len(x.size())==3: # we got a batch of inputs
            return torch.matmul(x, self.w.view(1,-1,1))
    
    def input_shape(self):
        return (1,len(self.w))


class DenseHead(nn.Module):
    def __init__(self, body,
                 body_out_dim=None,
                 out_dim = 1,
                 drop_rate =0.0,
                 activation=nn.Sigmoid(),
                 layers = 3):
        super(DenseHead, self).__init__()
        self.body = body
        self.dropout1 = nn.Dropout(drop_rate)
        hidden_dim = int(math.floor(body_out_dim/3)+1)
        self.dense_1 = nn.Linear(body_out_dim, 2*hidden_dim)
        self.dropout2 = nn.Dropout(drop_rate)
        self.dense_2 = nn.Linear(2*hidden_dim, hidden_dim)
        self.dropout3 = nn.Dropout(drop_rate)
        self.dense_3 = nn.Linear(hidden_dim, out_dim)
        self.activation = activation

    def forward(self, x):
        body_out = self.body(x)
        if type(body_out) == tuple:
            body_out = body_out[0]

        x1 =F.relu(self.dense_1(self.dropout1(body_out)))
        x2 = F.relu(self.dense_2(self.dropout2(x1)))
        x3 = self.activation(self.dense_3(self.dropout3(x2)))
        return x3