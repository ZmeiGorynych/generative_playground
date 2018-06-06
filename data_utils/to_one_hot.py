import torch
from torch.autograd import Variable
from basic_pytorch.gpu_utils import to_gpu, FloatTensor, LongTensor

def to_one_hot(y, n_dims=None, out = None):
    """
    Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims.
    The one-hot dimension is added at the end
    Taken from https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/24?u=egor_kraev
    """
    y_tensor = y.data if isinstance(y, Variable) else y
    y_tensor = y_tensor.type(LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims, out=out).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return Variable(y_one_hot) if isinstance(y, Variable) else y_one_hot