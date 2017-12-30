import torch
from math import floor, ceil
from utils import FloatTensor, IntTensor #
from torch import squeeze, unsqueeze


class WarpMatrix(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a_in):
        if len(a_in.shape)==2: # if just one sample rather than a batch
            # always use batch representation internally
            a_in = a_in.unsqueeze(0)
            ctx.squeeze = True
        else:
            ctx.squeeze = False
        ashape = list(a_in.shape)
        # process one slice at a time 
        dim1 = int(ashape[1])
        #t_dim = [ashape[0], dim1, dim1]
        trans_mat = FloatTensor(ashape[0], dim1, dim1).zero_()
        grad_indices = FloatTensor(ashape[0],dim1)
        cross_boundary = IntTensor(ashape[0],dim1)
        for n,a in enumerate(a_in):
            b = torch.cumsum(a, 0)     
            # dim0 = int(ceil(b[-1]))
            #t_dim = [dim0, dim1]
            #trans_grad = torch.zeros(t_dim.append(a.shape[0]))
            prev_ind = 0 
            for i, x in enumerate(zip(a, b)):
                ai, bi = x
                this_ind = floor(bi)
                if this_ind == prev_ind:
                    #print('ai',ai,'n',n,'this_ind', this_ind,'i',i)
                    trans_mat[n, this_ind, i] = ai[0]
                    cross_boundary[n,i] = False
                else:  # we just crossed an integer boundary
                    #print('ai',ai,'n',n,'this_ind', this_ind,'i',i)
                    tmp = bi - this_ind
                    trans_mat[n, this_ind, i] = tmp[0]
                    trans_mat[n, this_ind - 1, i] = ai[0] - tmp[0]
                    cross_boundary[n,i] = True
                grad_indices[n, i]=this_ind
                prev_ind = this_ind
        # assert ((a - trans_mat.sum(0)).abs().max() < 1e-6)
        # assert ((torch.ones(trans_mat.shape[0] - 1) - trans_mat.sum(1)[:-1]).abs().max() < 1e-6)
        #ctx.save_for_backward(a)
        ctx.a_in = a_in
        ctx.grad_indices = grad_indices
        ctx.cross_boundary = cross_boundary
        if ctx.squeeze:
            return squeeze(trans_mat,0)
        else:
            return trans_mat
    @staticmethod
    def backward(ctx,grad_output):
        if len(grad_output.shape)==2:
            grad_output = grad_output.unsqueeze(0)
        #print('grad output:', grad_output)
        #a, = ctx.saved_variables
        a_in = Variable(ctx.a_in)
        grad_indices = ctx.grad_indices
        my_grad = torch.zeros_like(a_in)
        for n,a in enumerate(a_in):
            for k, ind in enumerate(grad_indices[n]):
                my_grad[n, k] = grad_output[n,int(ind),k]
                for j in range(k+1,int(grad_output.data.shape[2])):
                    if ctx.cross_boundary[n,j]:
                        iofj = int(grad_indices[n,j])            
                        my_grad[n,k] = my_grad[n,k] +\
                                    (grad_output[n,iofj,j] - \
                                    grad_output[n,iofj-1,j] )
                    #print(my_grad[k].view(1,-1))
        if ctx.squeeze:
            return squeeze(my_grad,0) #torch.ones_like(a)
        else:
            return my_grad




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


class FittedWarp(torch.nn.Module):
    def __init__(self, w_shape= None, w= None):
        super().__init__()
        if w is not None:
            self.w = torch.nn.Parameter(to_gpu(w.data))
        else:
            self.w = torch.nn.Parameter(to_gpu(torch.randn(w_shape)))
        self.input_shape = [None,int(self.w.shape[0])]

    def forward(self, x):
        # x: n1 x n2, w: n2 x 1
        tmp1 = x @ self.w # n1 x 1
        tmp = torch.nn.Sigmoid()(tmp1) # n1 x 1
        trans_mat = WarpMatrix.apply(tmp) # n1 x n1
        #print(tmp, trans_mat)
        return trans_mat @ x # n1 x n2, but compressed over the first dim, so final rows are 0
 
from utils import to_gpu

class WarpMatrixOld(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a):
        b = torch.cumsum(a, 0)
        dim1 = int(a.shape[0])
        dim0 = int(ceil(b[-1]))
        #t_dim = [dim0, dim1]
        t_dim = [dim1, dim1]
        trans_mat = to_gpu(torch.zeros(t_dim))
        grad_indices = to_gpu(torch.FloatTensor(dim1))
        #trans_grad = torch.zeros(t_dim.append(a.shape[0]))
        prev_ind = 0
        cross_boundary = [None]*dim1
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
        #ctx.save_for_backward(a)
        ctx.a = a
        ctx.grad_indices = grad_indices
        ctx.cross_boundary = cross_boundary
        return to_gpu(trans_mat)
    @staticmethod
    def backward(ctx,grad_output):
        #print('grad output:', grad_output)
        #a, = ctx.saved_variables
        a = Variable(ctx.a)
        grad_indices = ctx.grad_indices
        my_grad = to_gpu(torch.zeros_like(a))
        for k, ind in enumerate(grad_indices):
            my_grad[k] = grad_output[int(ind),k]
            for j in range(k+1,int(grad_output.data.shape[1])):
                if ctx.cross_boundary[j]:
                    iofj = int(grad_indices[j])            
                    my_grad[k] = my_grad[k] +\
                                (grad_output[iofj,j] - \
                                grad_output[iofj-1,j] )
                    #print(my_grad[k].view(1,-1))
        return my_grad #torch.ones_like(a)



class FittedWarpOld(torch.nn.Module):
    def __init__(self, w_shape):
        super().__init__()
        self.w = torch.nn.Parameter(to_gpu(torch.randn(w_shape)))
        self.input_shape = [None,int(self.w.shape[0])]

    def forward(self, x):
        # x: n1 x n2, w: n2 x 1
        tmp1 = x @ self.w # n1 x 1
        tmp = torch.nn.Sigmoid()(tmp1) # n1 x 1
        trans_mat = WarpMatrixOld.apply(tmp) # n1 x n1
        #print(tmp, trans_mat)
        return trans_mat @ x # n1 x n2, but compressed over the first dim, so final rows 

if __name__ == '__main__':
    dim = 6
    a = Variable(torch.randn([10, dim]))
    warp = FittedWarp(dim)
    y = warp(a)

    #print(y.requires_grad)