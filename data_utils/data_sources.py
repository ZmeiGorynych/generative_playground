import torch
import h5py
import math
import numpy as np
from torch.autograd import Variable
from basic_pytorch.gpu_utils import to_gpu, use_gpu
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import SubsetRandomSampler

class DatasetFromHDF5(Dataset):
    def __init__(self, filename, dataset, len_dim=0):
        h5f = h5py.File(filename, 'r')
        self.data = h5f[dataset]
        if not len_dim==0:
            raise NotImplementedError("Don't support other batch dimensions than 0 just yet")
        self.len_dim = len_dim
        
    def __len__(self):
        return self.data.shape[self.len_dim]

    def __getitem__(self, item):
        return self.data[item].astype(float)

def train_valid_loaders(dataset, valid_fraction =0.1, **kwargs):
    # num_workers
    # batch_size
    num_train = len(dataset)
    indices = list(range(num_train))
    split = int(math.floor(valid_fraction* num_train))

    if not('shuffle' in kwargs and not kwargs['shuffle']):
            #np.random.seed(random_seed)
            np.random.shuffle(indices)
    if 'num_workers' not in kwargs:
        kwargs['num_workers'] = 1

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(dataset,
                                               sampler=train_sampler,
                                               **kwargs)
    valid_loader = torch.utils.data.DataLoader(dataset,
                                               sampler=valid_sampler,
                                               **kwargs)

    return train_loader, valid_loader

class DatasetFromModel(Dataset):
    """Dataset creating data on the fly"""

    def __init__(self, first_dim, batches, model):
        self.batches = batches
        self.model = model
        self.x_shape = first_dim, model.input_shape()[1]
        

    def __len__(self):
        return self.batches

    def __getitem__(self, idx):
        if idx<self.batches:
            x = to_gpu(torch.randn(self.x_shape))
            y = to_gpu(self.model(Variable(x)).data)
            return (x, y)
        else:
            raise StopIteration()