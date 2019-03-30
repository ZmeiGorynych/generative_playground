from collections import OrderedDict
from torch import nn as nn
from torch.nn import functional as F

from generative_playground.utils.gpu_utils import to_gpu



class MultipleOutputHead(nn.Module):
    def __init__(self, model, output_spec, drop_rate=0.2):
        '''
        Takes a model that outputs an array of Floats and does a bunch of linear transforms on it
        # TODO: should we be including a relu here?
        :param model: The upstream model whose output we're processing
        :param output_dims: a list or a dict, values are either ints or modules
        :param drop_rate:
        :param labels:
        '''
        super().__init__()
        self.model = model
        self.dropout = nn.Dropout(drop_rate)
        self.output_spec = output_spec
        if isinstance(output_spec, list) or isinstance(output_spec, tuple):
            module_list =[to_gpu(self._get_module(s)) for s in self.output_spec]
            self.fcs = nn.ModuleList(module_list)
            self.fcs_dict = None
            self.output_shape = [model.output_shape[:-1] + [s] for s in self.sizes]
        else: # assume output_spec is a dict
            module_dict = {key: self._get_module(val) for key,val in output_spec.items()}
            self.fcs_dict = nn.ModuleDict(module_dict)
            self.fcs = None
            self.output_shape = {label: model.output_shape[:-1] + [s] for label, s in self.output_spec.items()}

    def _get_module(self, spec):
        if isinstance(spec, nn.Module):
            return spec
        else:
            return nn.Linear(self.model.output_shape[-1], spec)

    def forward(self, x):
        '''

        :param x: either batch x num_dims or batch x num_steps x num_dims
        :return: len(self.sizes) vectors of size x.size()[:2] x self.sizes[i],
        as tuple if self.labels is None, as dict otherwise
        '''
        out = F.relu(self.model.forward(x))
        out = self.dropout(out)

        if self.fcs is None:
            out = {label: fc(out) for label, fc in self.fcs_dict.items()}
        else:
            out = [fc(out) for fc in self.fcs]

        return out

    # def init_encoder_output(self, z):
    #     '''
    #     Must be called at the start of each new sequence
    #     :param z: encoder output
    #     :return: None
    #     '''
    #     assert hasattr(self.model,'init_encoder_output'), "The underlying model is missing the init_encoder_output method"
    #     self.model.init_encoder_output(z)