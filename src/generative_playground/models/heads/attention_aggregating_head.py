from torch import nn as nn
from torch.nn import functional as F


class AttentionAggregatingHead(nn.Module):
    def __init__(self, model, drop_rate=0.1):
        super().__init__()
        self.model = model
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
        if isinstance(model_out, tuple): # some models return extra info, which we now discard
            model_out=model_out[0]
        assert(len(model_out.size())==3, 'AttentionAggregatingHead needs sequences as inputs')
        model_out = self.dropout_1(model_out.contiguous())
        batch_size = model_out.size()[0]
        # calculate attention weights
        pre_attn_wgt = self.attention_wgt(model_out.view(-1, model_out.size()[-1]))
        attn_wgt = F.softmax(pre_attn_wgt.view(batch_size, model_out.size()[1], 1), dim=1)
        # and use them to aggregate
        out = (attn_wgt * self.dropout_2(model_out)).sum(1)
        return out