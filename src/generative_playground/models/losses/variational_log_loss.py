import torch.nn as nn
import torch

class VariationalLoss(nn.Module):
    def __init__(self, labels, use_skew = False):
        super().__init__()
        self.metrics ={}
        self.labels = labels
    def forward(self, model_out, target_x):
        '''

        :param model_out: mean, logvar of same size as target_x
        :param target_x:
        :return:
        '''
        loss, mean = normal_ll(target_x, model_out)

        valid = target_x[:, 0] == 1  # convention is that the first col is molecule validity
        valid_loss = loss[:,0]
        # all other metrics' losses are conditional on molecule being valid
        other_loss = loss[valid, 1:]

        self.metrics = self.get_metrics(target_x, valid, mean, model_out)

        return torch.mean(valid_loss,0) + torch.mean(other_loss,0).sum()


    def get_metrics(self, target_x, valid, mean, model_out):
        metrics ={}
        dist_mean, logvar, skew = model_out
        err = (mean - target_x)

        metrics['pct_valid'] = torch.mean(target_x[:, 0]).data.item()
        for i in range(target_x.size()[-1]):
            if i == 0:
                this_err = err[:, 0]
            else:
                this_err = err[valid, i]
            avg_err = torch.mean(torch.abs(this_err), 0)
            avg_std = torch.mean(torch.exp(0.5 * logvar[:, i]), 0)
            metrics['avg err ' + self.labels[i]] = avg_err.data.item()  # avg_err[i].data.item()
            metrics['avg std ' + self.labels[i]] = avg_std.data.item()  # avg_std[i].data.item()
        return metrics

def normal_ll(target_x, model_out):
    mean, logvar, skew = model_out
    err = (mean - target_x)
    loss = err*err/torch.exp(logvar) + logvar

    return loss, mean

