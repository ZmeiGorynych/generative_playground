import torch
from torch import nn as nn
from torch.nn import functional as F
from generative_playground.models.transformer.SubLayers import MultiHeadAttention
from generative_playground.utils.gpu_utils import device

class AttentionAggregatingHead(nn.Module):
    def __init__(self, model, drop_rate=0.1):
        super().__init__()
        self.model = model
        self.model_out_transform = lambda x: x[0] if isinstance(x, tuple) else x
        feature_len = model.output_shape[-1]
        self.attention_wgt = nn.Linear(feature_len,1)
        self.dropout_1 = nn.Dropout(drop_rate)
        self.dropout_2 = nn.Dropout(drop_rate)
        self.output_shape = [None, feature_len]

    def forward(self, x):
        '''
        Return an attetion-based aggregation of a sequence
        :param x: batch x seq_len x feature_len
        :return: batch x feature_len
        '''
        model_out = self.model(x)
        model_out = self.model_out_transform(model_out)

        assert(len(model_out.size())==3, 'AttentionAggregatingHead needs sequences as inputs')
        model_out = self.dropout_1(model_out.contiguous())
        batch_size = model_out.size()[0]
        # calculate attention weights
        pre_attn_wgt = self.attention_wgt(model_out.view(-1, model_out.size()[-1]))
        attn_wgt = F.softmax(pre_attn_wgt.view(batch_size, model_out.size()[1], 1), dim=1)
        # and use them to aggregate
        out = (attn_wgt * self.dropout_2(model_out)).sum(1)
        return out


class MultiheadAttentionAggregatingHead(nn.Module):
    def __init__(self, model, drop_rate=0.1, n_head=6, d_k=64, d_v=64):
        super().__init__()
        self.model = model.to(device)
        self.attention = MultiHeadAttention(n_head=n_head,
                                            d_model=self.model.output_shape[-1],
                                            d_k=d_k,
                                            d_v=d_v,
                                            dropout=drop_rate
                                            ).to(device)
        self.key = torch.zeros(1,
                                  1,
                                  self.model.output_shape[-1],
                                  dtype=torch.float32,
                                  device=device,
                                  requires_grad=True)
        nn.init.xavier_uniform_(self.key)
        self.output_shape = [model.output_shape[0], model.output_shape[2]]

    def forward(self, x):
        # TODO: propagate masks too?
        model_out = self.model(x)
        keys = self.key.expand(model_out.size(0),1,-1)
        out, _ = self.attention(keys, model_out, model_out, attn_mask=None)
        return out[:,0,:]

class FirstSequenceElementHead(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.output_shape = [model.output_shape[0], model.output_shape[2]]

    def forward(self, *input):
        out = self.model(*input)
        return out[:,0,:]
