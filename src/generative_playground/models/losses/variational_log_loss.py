import torch.nn as nn
import torch

class VariationalLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.metrics ={}
    def forward(self, model_out, target_x):
        mean, logvar = model_out
        err = (mean - target_x)
        loss = err*err/torch.exp(logvar) + logvar
        avg_err = torch.mean(torch.abs(err),0)
        avg_std = torch.mean(torch.exp(0.5 * logvar),0)
        self.metrics['pct_valid'] = torch.mean(target_x[:,0]).data.item()
        for i in range(len(avg_err)):
            self.metrics['avg_err ' + str(i) ] = avg_err[i].data.item()
            self.metrics['avg std' + str(i) ] = avg_std[i].data.item()
        return torch.mean(loss,0).sum()