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
        works around hdf5's problems with simultaneous reads and writes
        :param fname:
        :param valid_frac:
        '''
        self.fname = fname
        self.valid_frac = valid_frac
        self.h5f = None
        self.data = None
        self.idx  = None
        self.init_read_handle()
        self.datasets_created = False

    def init_read_handle(self):
        try:
            if self.h5f is None:
                self.h5f = h5py.File(self.fname, 'a')

            if self.data is None:
                try:
                    self.data = self.h5f['data']
                except Exception as e:
                    pass
            if self.valid_frac is not None and self.idx is None:
                try:
                    self.idx = {True: self.h5f['valid_idx'],
                                False: self.h5f['train_idx']}
                except:
                    pass
        except Exception as e:
            print('Read open failed. Maybe you need to do some appends first?')

    def __len__(self):
        return self.data.shape[self.len_dim]

    def __getitem__(self, item):
        return self.data[item].astype(float)

    def append(self, data):
        '''
        Append a new slice of data, assume that data.shape[1:] never changes
        :param valid:
        :param data:
        :return:
        '''
        if len(data)==0:
            return

        try:
            base_len = len(self.h5f["data"])
        except: # if there is no such dataset yet
            base_len = 0

        # TODO: replace this dirty hack with an nicer workaround
        if self.valid_frac is not None and base_len == 0 and len(data)<2:
            #raise ValueError("If valid_frac is set, first data batch must have at least 2 elements")
            data=np.array([data[0],data[0]])
        # close the read handle
        # if self.h5f_read is not None:
         #   self.h5f_read.close()
        #with h5py.File(self.fname, 'a') as self.h5f:


        try: # if dataset already created
            self.h5f["data"].resize(self.h5f["data"].shape[0] + data.shape[0], axis=0)
            self.h5f["data"][-data.shape[0]:] = data
        except Exception as e:
            if len(data.shape)==1:
                ds_dim = [None]
            else:
                ds_dim = [None] + list(data.shape[1:])
            self.h5f.create_dataset('data', data=data,
                               compression="gzip",
                               compression_opts=9,
                               maxshape=ds_dim)

        if self.valid_frac is not None:
            # randomly assign data to valid/train subsets
            is_valid = np.array([self.valid_frac >= np.random.uniform(size=len(data))])[0]
            if base_len == 0: # want to guarantee both index sets are nonempty
                is_valid[0] = True
                is_valid[1] = False

            all_ind = base_len + np.array(range(len(data)))
            valid_ind = all_ind[is_valid]
            train_ind = all_ind[is_valid == False]
            for data_, ds_name in (valid_ind,'valid_idx'), (train_ind,'train_idx'):
                if not len(data_):
                    continue
                try: # if dataset already created
                    self.h5f[ds_name].resize(self.h5f[ds_name].shape[0] + len(data_), axis=0)
                    self.h5f[ds_name][-data_.shape[0]:] = data_
                except Exception as e:
                    print(e)
                    self.h5f.create_dataset(ds_name, data=data_,
                                       compression="gzip",
                                       compression_opts=9,
                                       maxshape=tuple([None]))
        # refresh the read handle
        self.init_read_handle()

    def get_item(self, item, valid):
        return self.data[self.idx[valid][item]]

    def get_len(self, valid):
        return len(self.idx[valid])

    def get_train_valid_loaders(self, batch_size):
        train_ds = ChildHDF5Dataset(self, False)
        val_ds = ChildHDF5Dataset(self, True)
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
    def __init__(self, parent, valid):
        self.parent = parent
        self.valid = valid

    def __len__(self):
        return len(self.parent.idx[self.valid])

    def __getitem__(self, item):
        if item < self.__len__():
            return self.parent.data[self.parent.idx[self.valid][item]]
        else:
            raise ValueError("Item exceeds dataset length")