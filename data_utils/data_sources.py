import torch
import h5py
import math
import numpy as np
from torch.autograd import Variable
from basic_pytorch.gpu_utils import to_gpu, use_gpu
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import SubsetRandomSampler, RandomSampler

class DatasetFromHDF5(Dataset):
    '''
    A simple Dataset wrapper around an hdf5 file
    '''
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
    # if 'num_workers' not in kwargs:
    #     kwargs['num_workers'] = 1

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


class DuplicateIter:
    def __init__(self, iterable):
        self.iterable = iterable

    def __iter__(self):
        def gen():
            iter = self.iterable.__iter__()
            while True:
                # TODO: cast to float earlier?
                x = to_gpu(next(iter).float())
                yield (x, x)

        return gen()

    def __len__(self):
        return len(self.iterable)


class IncrementingHDF5Dataset:
    def __init__(self, fname, valid_frac = None):
        '''
        An hdf5 wrapper that can be incremented on the fly,
        works around hdf5's problems with simultaneous reads and writes,
        assigning slices to train/validation on the fly
        :param fname: hdf5 filename
        :param valid_frac: fraction of validation samples
        '''
        self.fname = fname
        self.valid_frac = valid_frac
        self.h5f = None
        self.dataset_names = set()
        self.idx = {True: None, False: None}
        self.h5f = h5py.File(self.fname, 'a')

    def __len__(self):
        if len(self.dataset_names) == 0:
            return 0
        else:
            return len(self.h5f[self.dataset_names[0]])

    # def __getitem__(self, item):
    #     return self.data[item].astype(float)
    # TODO: does thiw work with lists of strings, FloatTensors, LongTensors?
    def append_to_dataset(self, dataset_name, data):
        if len(data)==0:
            return

        try:
            self.h5f[dataset_name].resize(self.h5f[dataset_name].shape[0] + data.shape[0], axis=0)
            self.h5f[dataset_name][-data.shape[0]:] = data
        except: # if there is no such dataset yet
            if len(data.shape)==1:
                ds_dim = [None]
            else:
                ds_dim = [None] + list(data.shape[1:])
            self.h5f.create_dataset(dataset_name, data=data,
                               compression="gzip",
                               compression_opts=9,
                               maxshape=ds_dim)
            self.dataset_names.add(dataset_name)

    def append(self, data):
        '''
        Append a new slice of data, assume that data.shape[1:] never changes
        :param valid:
        :param data:
        :return:
        '''
        if len(data)==0:
            return

        if type(data) != dict:
            return self.append({'data':data})

        a_dataset_name = list(data.keys())[0]


        try:
            base_len = len(self.h5f[a_dataset_name])
        except: # if there is no such dataset yet
            base_len = 0

        new_len = None
        for ds_name, new_data in data.items():
            # all new data chunks must have the same length for train/valid indices to work
            if new_len == None:
                new_len = len(new_data)
            else:
                assert(new_len == len(new_data))
            self.append_to_dataset(ds_name, new_data)

        if self.valid_frac is not None:
            # randomly assign new data to valid/train subsets
            is_valid = np.array([self.valid_frac >= np.random.uniform(size=new_len)])[0]

            all_ind = base_len + np.array(range(new_len))
            valid_ind = all_ind[is_valid]
            train_ind = all_ind[is_valid == False]
            for data_, ds_name, valid in (valid_ind,'valid_idx',True), \
                                         (train_ind,'train_idx', False):
                if not len(data_):
                    continue
                else:
                    self.append_to_dataset(ds_name, data_)
                    self.idx[valid] = self.h5f[ds_name]

        # refresh the read handle
        # self.init_read_handle()

    def get_item(self, dataset, item, valid=None):
        if valid is None:
            return self.h5f[dataset][item]
        elif self.idx[valid] is not None:
            return self.h5f[dataset][self.idx[valid][item]]
        else:
            return None

    def get_len(self, valid):
        if valid is None:
            return self.__len__()
        elif self.idx[valid] is not None:
            return len(self.idx[valid])
        else:
            return 0

    def get_train_valid_loaders(self, batch_size, dataset_name='data'):
        train_ds = ChildHDF5Dataset(self,
                                    valid=False,
                                    dataset_name=dataset_name)
        val_ds = ChildHDF5Dataset(self,
                                  valid=True,
                                  dataset_name=dataset_name)
        train_loader = torch.utils.data.DataLoader(train_ds,
                                                   sampler=RandomSampler(train_ds),
                                                   batch_size=batch_size,
                                                   pin_memory=use_gpu)
        valid_loader = torch.utils.data.DataLoader(val_ds,
                                                   sampler=RandomSampler(val_ds),
                                                   batch_size=batch_size,
                                                   pin_memory=use_gpu)

        return train_loader, valid_loader


class ChildHDF5Dataset:
    def __init__(self, parent, dataset_name='data', valid=None):
        self.parent = parent
        self.valid = valid
        self.dataset_name = dataset_name

    def __len__(self):
        return self.parent.get_len(self.valid)

    def __getitem__(self, item):
        if item < self.__len__():
            # if we got this far, self.__len__() is >0 so we have indices
            if type(self.dataset_name) == str:
                return self.parent.get_item(self.dataset_name,
                                            self.valid,
                                            self.item)
            elif type(self.dataset_name) in (tuple, list):
                out = tuple(self.parent.get_item(dsname,
                                            self.valid,
                                            self.item) for dsname in self.dataset_name)
                return out
        else:
            raise ValueError("Item exceeds dataset length")