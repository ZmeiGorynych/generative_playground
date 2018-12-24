import torch
from torch import nn as nn
from torch.autograd import Variable

from generative_playground.models.decoder.policy import PolicyFromTarget
from generative_playground.models.heads.mean_variance_head import MeanVarianceSkewHead
from generative_playground.utils.gpu_utils import to_gpu, FloatTensor


class VariationalAutoEncoderHead(nn.Module):
    def __init__(self, encoder=None,
                 decoder=None,
                 sample_z=True,
                 epsilon_std=0.01,
                 z_size = None):
        '''
        Initialize the autoencoder
        :param encoder: A model mapping batches of one-hot sequences (batch x seq x num_actions) to batches of logits
        :param decoder: Model mapping latent z (batch x z_size) to  batches of one-hot sequences, and corresponding logits
        :param sample_z: Whether to sample z = N(mu, std) or just take z=mu
        :param epsilon_std: Scaling factor for samling, low values help convergence
        https://github.com/mkusner/grammarVAE/issues/7
        '''
        super(VariationalAutoEncoderHead, self).__init__()
        self.sample_z = sample_z
        self.encoder = to_gpu(encoder)
        self.decoder = to_gpu(decoder)
        self.epsilon_std = epsilon_std
        self.mu_var_layer = to_gpu(MeanVarianceSkewHead(self.encoder, z_size))
        self.output_shape = [None, z_size]

    def forward(self, x):
        dist = self.mu_var_layer(x)
        mu = dist[0]
        log_var = dist[1]

        # only sample when training, I regard sampling as a regularization technique so unneeded during validation
        if self.sample_z and self.training:
            z = self.sample(mu, log_var)
        else:
            z = mu
        # need to re-trace the path taken by the training input
        # this also takes care of masking
        _, x_actions = torch.max(x,-1) # argmax
        self.decoder.policy = PolicyFromTarget(x_actions)
        actions, output = self.decoder(z)
        return output, mu, log_var

    def sample(self, mu, log_var):
        """you generate a random distribution w.r.t. the mu and log_var from the embedding space."""
        vector_size = log_var.size()
        eps = self.epsilon_std*Variable(FloatTensor(vector_size).normal_())
        std = log_var.mul(0.5).exp_()
        return eps.mul(std).add_(mu)

    def load(self, weights_file):
        print('Trying to load model parameters from ', weights_file)
        self.load_state_dict(torch.load(weights_file))
        self.eval()
        print('Success!')