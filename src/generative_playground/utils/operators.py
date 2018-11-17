import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence
from torch.autograd.gradcheck import gradgradcheck, gradcheck


"""
A selection of differentiable 'max' operators
Adapted from https://github.com/arthurmensch/didyprog

Author: Egor Kraev
"""

class HardMaxOp:
    @staticmethod
    def max(X):
        M, _ = torch.max(X, dim=-1, keepdim=True)
        A = (M == X).float()
        A = A / torch.sum(A, dim=-1, keepdim=True)

        return M.squeeze(), A#.squeeze()

    @staticmethod
    def hessian_product(P, Z):
        return torch.zeros_like(Z)


class SoftMaxOp:
    @staticmethod
    def max(X):
        M, _ = torch.max(X, dim=-1)
        X = X - M.unsqueeze(len(M.size()))#(M[:, :, None]
        A = torch.exp(X)
        S = torch.sum(A, dim=-1)
        M = M + torch.log(S)
        A /= S.unsqueeze(len(S.size()))#S[:, :, None]
        return M.squeeze(), A#.squeeze()


    @staticmethod
    def hessian_product(P, Z):
        prod = P * Z
        return prod - P * torch.sum(prod, dim=-1, keepdim=True)


class SparseMaxOp:
    @staticmethod
    def max(X):
        if len(X.shape) ==1:
            X = X.unsqueeze(0)
            squeeze_back = True
        else:
            squeeze_back = False
        n_states = X.shape[-1]
        other_dims = X.shape[:-1]
        X_sorted, _ = torch.sort(X, dim=-1, descending=True)
        cssv = torch.cumsum(X_sorted, dim=-1) - 1
        ind = X.new(n_states)
        for i in range(n_states):
            ind[i] = i + 1
        cond = X_sorted - cssv / ind > 0
        rho = cond.long().sum(dim=-1)
        cssv = cssv.view(-1, n_states)
        rho = rho.view(-1)

        tau = (torch.gather(cssv, dim=1, index=rho[:, None] - 1)[:, 0]
               / rho.type(X.type()))
        tau = tau.view(*other_dims)
        A = torch.clamp(X - tau.unsqueeze(len(tau.size())), min=0)
        # A /= A.sum(dim=2, keepdim=True)

        M = torch.sum(A * (X - .5 * A), dim=-1)
        if squeeze_back:
            A = A.squeeze(0)

        return M.squeeze(), A#M.squeeze(), A.squeeze()

    @staticmethod
    def hessian_product(P, Z):
        S = (P > 0).type(Z.type())
        support = torch.sum(S, dim=-1, keepdim=True)
        prod = S * Z
        return prod - S * torch.sum(prod, dim=-1, keepdim=True) / support

operators = {'softmax': SoftMaxOp, 'sparsemax': SparseMaxOp,
             'hardmax': HardMaxOp}

def make_custom_max_function(type):
    assert(type in operators, "Unknown type " + type + ". Must be softmax, hardmax, or sparsemax")
    this_op = operators[type]
    class CustomMaxFunction(torch.autograd.Function):
        """
        Custom max function using the chosen operator
        TODO: custom second derivative to speed things up
        """
        @staticmethod
        def forward(ctx, X):
            M, A = this_op.max(X)
            ctx.save_for_backward(A)
            return M

        @staticmethod
        def backward(ctx, M):
            A, = ctx.saved_tensors
            if len(M.size()) == 0:
                return M*A
            else:
                return torch.mm(M, A)

    return CustomMaxFunction

class CustomMax(nn.Module):
    def __init__(self, max_type):
        super().__init__()
        self.max_op = make_custom_max_function(max_type)

    def forward(self, X):
        return self.max_op.apply(X)


for type in ['hardmax','softmax','sparsemax']:
    # TODO: make sparsemax work for dimension 0
    for num_dims in range(1,6):
        pre_x = [-10,2,2.1]
        for _ in range(num_dims-1):
            pre_x = [pre_x]
        X = torch.Tensor(pre_x)
        X.requires_grad_()
        print('x:', X)
        # truemax = torch.max(X)
        # truemax.backward()
        # print(X.grad)

        mymaxfun = CustomMax(type)
        mymax = mymaxfun(X)
        print('mymax(x):', mymax)
        mymax.backward()
        print(X.grad)

        X.grad.zero_()
        gradcheck(mymaxfun, (X, ), eps=1e-4, atol=1e-2)
