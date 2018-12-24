from torch import nn as nn
from torch.nn import functional as F

from generative_playground.utils.gpu_utils import to_gpu


class MultipleOutputHead(nn.Module):
    def __init__(self, model, output_dims, drop_rate=0.2, labels=None):
        super().__init__()
        self.labels = labels
        try:
            iter(output_dims)
        except: # if it was just a number
            output_dims =(output_dims)
        self.sizes = output_dims
        self.model = model
        self.dropout = nn.Dropout(drop_rate)
        module_list =[to_gpu(nn.Linear(self.model.output_shape[-1], s)) for s in self.sizes]
        self.fcs = nn.ModuleList(module_list)
        if labels is None:
            self.output_shape = [model.output_shape[:-1] + [s] for s in self.sizes]
        else:
            self.output_shape = {label: model.output_shape[:-1] + [s] for label, s in zip(labels,self.sizes)}

    def forward(self, x):
        '''

        :param x: either batch x num_dims or batch x num_steps x num_dims
        :return: len(self.sizes) vectors of size x.size()[:2] x self.sizes[i],
        as tuple if self.labels is None, as dict otherwise
        '''
        out = F.relu(self.model.forward(x))
        out = self.dropout(out)
        if self.labels:
            out = {label: fc(out) for label, fc in zip(self.labels, self.fcs)}
        else:
            out = [fc(out) for fc in self.fcs]
        return out