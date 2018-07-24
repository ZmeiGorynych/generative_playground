from collections.__init__ import OrderedDict

import torch
from torch import nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

from generative_playground.utils.gpu_utils import to_gpu


class VAELoss(nn.Module):
    # matches the impelentation in model_eq.py
    def __init__(self, grammar=None, sample_z=False, reg_weight = 0.01):
        '''
        :param masks: array of allowed transition rules from a given symbol
        '''
        super(VAELoss, self).__init__()
        self.sample_z = sample_z
        self.bce_loss = nn.BCELoss(size_average = True) #following mkusner/grammarVAE
        self.reg_weight = reg_weight
        # if grammar is not None:
        #     self.masks = FloatTensor(grammar.masks)
        #     self.ind_to_lhs_ind = LongTensor(grammar.ind_to_lhs_ind)
        # else:
        #     self.masks = None

    def forward(self, model_out, target_x):
        """gives the batch normalized Variational Error."""
        model_out_x, mu, log_var = model_out
        batch_size = target_x.size()[0]
        seq_len = target_x.size()[1]
        z_size = mu.size()[1]
        model_out_x = F.softmax(model_out_x, dim=2)
        #following mkusner/grammarVAE
        BCE = seq_len * self.bce_loss(model_out_x, target_x)
        # this normalizer is for when we're not sampling so only have mus, not sigmas
        avg_mu = torch.sum(mu, dim=0) / batch_size
        var = torch.mm(mu.t(), mu) / batch_size
        var_err = var - Variable(to_gpu(torch.eye(z_size)))
        var_err = F.tanh(var_err)*var_err # so it's ~ x^2 asymptotically, not x^4
        mom_err = (avg_mu * avg_mu).sum() / z_size + var_err.sum() / (z_size * z_size)
        if self.sample_z:
            # see Appendix B from VAE paper:
            # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
            # https://arxiv.org/abs/1312.6114
            # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
            KLD_element = (1 + log_var - mu*mu - log_var.exp())
            KLD = -0.5 * torch.mean(KLD_element)
            KLD_ = KLD.data.item()
            my_loss = BCE + self.reg_weight * KLD
        else:
            my_loss = BCE + self.reg_weight * mom_err
            KLD_ = 0
        if not self.training:
            # ignore regularizers when computing validation loss
            my_loss = BCE

        self.metrics = OrderedDict([('BCE', BCE.data.item()),
                                   ('KLD', KLD_),
                                   ('ME', mom_err.data.item())])
        return my_loss