import torch
from math import floor, ceil


class WarpMatrix(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a):
        b = torch.cumsum(a, 0)
        dim1 = int(a.shape[0])
        dim0 = int(ceil(b[-1]))
        #t_dim = [dim0, dim1]
        t_dim = [dim1, dim1]
        trans_mat = torch.zeros(t_dim)
        grad_indices = torch.FloatTensor(dim1)
        #trans_grad = torch.zeros(t_dim.append(a.shape[0]))
        prev_ind = 0
        cross_boundary = []
        for i, x in enumerate(zip(a, b)):
            ai, bi = x
            this_ind = floor(bi)
            if this_ind == prev_ind:
                trans_mat[this_ind, i] = ai[0]
                cross_boundary.append(False)
            else:  # we just crossed an integer boundary
                tmp = bi - this_ind
                trans_mat[this_ind, i] = tmp[0]
                trans_mat[this_ind - 1, i] = ai[0] - tmp[0]
                cross_boundary.append(True)
            grad_indices[i]=this_ind
            prev_ind = this_ind
        # assert ((a - trans_mat.sum(0)).abs().max() < 1e-6)
        # assert ((torch.ones(trans_mat.shape[0] - 1) - trans_mat.sum(1)[:-1]).abs().max() < 1e-6)
        ctx.save_for_backward(a)
        ctx.grad_indices = grad_indices
        ctx.cross_boundary = cross_boundary
        return trans_mat
    @staticmethod
    def backward(ctx,grad_output):
        #print('grad output:', grad_output)
        a, = ctx.saved_variables
        grad_indices = ctx.grad_indices
        my_grad = torch.zeros_like(a)
        for k, ind in enumerate(grad_indices):
            my_grad[k] = grad_output[int(ind),k]
            for j in range(k+1,int(grad_output.data.shape[1])):
                if ctx.cross_boundary[j]:
                    iofj = int(grad_indices[j])            
                    my_grad[k] = my_grad[k] +\
                                (grad_output[iofj,j] - \
                                grad_output[iofj-1,j] )
                    #print(my_grad[k].view(1,-1))
        return my_grad#torch.ones_like(a)




# class WarpMatrix(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, x):
#         return warp_matrix(x)


# a = torch.FloatTensor(10).uniform_()
# print(a)
# trans_mat = warp_matrix(a)
# print(trans_mat.sum(1))

from torch.autograd import Variable


class FittedWarp__(torch.nn.Module):
    def __init__(self, w_shape):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(w_shape))

    def forward(self, x):
        tmp1 = x @ self.w
        tmp = torch.nn.Sigmoid()(tmp1)
        trans_mat = warp_matrix(tmp)
        #print(tmp, trans_mat)
        return trans_mat @ x


if __name__ == '__main__':
    dim = 6
    a = Variable(torch.randn([10, dim]))
    warp = FittedWarp(dim)
    y = warp(a)

    #print(y.requires_grad)