import torch.nn as nn
import torch

def collect_fcs(model_out, target, str_to_int):
    '''

    :param model_out: {0: float_tensor, 1: float_tensor, ...}
    :param target:
    :return:
    '''
    embeds = torch.zeros(target.size()[0], target.size()[1], model_out['1'].size()[2],
                         dtype=torch.float32, device=target.device)
    langs = target[:,0]
    for key in model_out.keys():
        if key in str_to_int and str_to_int[key] in langs:
            embeds[langs == str_to_int[key], :, :] = model_out[key][langs == str_to_int[key], :, :]

    return embeds

class MultipleCrossEntropyLoss(nn.Module):
    def __init__(self, multi_language=None):
        super().__init__()
        self.metrics ={}
        self.celoss = nn.CrossEntropyLoss(size_average=True)#,
                                          #ignore_index=0)
        self.multi_language = multi_language
        self.str_to_int = {str(i): i for i in range(20)}

    def forward(self, model_out, target_x):
        '''
        Calculate cross-entropy on a number of targets at once
        :param model_out: a dict of {string: batch_size x maxlen x num_features float tensors} (num_features differs),
        :param target_x: a dict of batch_size x maxlen longs, len(target_x) <= len(model_out)
        :return:
        '''
        if self.multi_language is not None:
            model_out[self.multi_language] = collect_fcs(model_out,
                                                         target_x[self.multi_language],
                                                         self.str_to_int)
        loss = 0
        for label, tgt in target_x.items():
            input = model_out[label].transpose(1, 2) # CrossEntropyLoss wants the one-hot dim to be 1
            this_loss = self.celoss(input, tgt)
            self.metrics[label] = this_loss.item()
            loss += this_loss

        return loss


#
#     def get_metrics(self, target_x, valid, mean, model_out):
#         metrics ={}
#         dist_mean, logvar, skew = model_out
#         err = (mean - target_x)
#
#         metrics['pct_valid'] = torch.mean(target_x[:, 0]).data.item()
#         for i in range(target_x.size()[-1]):
#             if i == 0:
#                 this_err = err[:, 0]
#             else:
#                 this_err = err[valid, i]
#             avg_err = torch.mean(torch.abs(this_err), 0)
#             avg_std = torch.mean(torch.exp(0.5 * logvar[:, i]), 0)
#             metrics['avg err ' + self.labels[i]] = avg_err.data.item()  # avg_err[i].data.item()
#             metrics['avg std ' + self.labels[i]] = avg_std.data.item()  # avg_std[i].data.item()
#         return metrics
#
# def normal_ll(target_x, model_out):
#     mean, logvar, skew = model_out
#     err = (mean - target_x)
#     loss = err*err/torch.exp(logvar) + logvar
#
#     return loss, mean

