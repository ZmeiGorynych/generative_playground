import torch
import h5py
import math
import numpy as np
from torch.autograd import Variable
from gpu_utils import to_gpu, use_gpu
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
# fit_dataset = ImagesDataset(path_to_fit_images)
# fit_data_loader = torch.utils.data.DataLoader(fitdataset, batch_size=10)

# def data_gen(batch_size, batches, model):
#     for _ in range(batches):
#         x = to_gpu(Variable(torch.randn(batch_size, model.input_shape()[1])))
#         y = to_gpu(model.forward(x))
#         y = y.detach()
#         yield x, y

class DatasetFromHDF5(Dataset):
    def __init__(self, filename, dataset, len_dim=0):
        h5f = h5py.File(filename, 'r')
        self.data = h5f[dataset]
        if not len_dim==0:
            raise NotImplementedError("Don't support other batch dimensions than 2 just yet")
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