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
    def __init__(self, filename, dataset):
        h5f = h5py.File(filename, 'r')
        self.data = h5f[dataset]

    def __len__(self):
        return len(self.data)

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
    def __init__(self, fname, mode='a'):#
        '''
        An hdf5 wrapper that can be incremented on the fly,
        works around hdf5's problems with simultaneous reads and writes,
        assigning slices to train/validation on the fly
        :param fname: hdf5 filename
        :param mode: 'a' if you also want to append to the file while training,
        'r' if you want to allow multiple processes to open the file
        sadly if one process has the file open for writing, no other process can open it
        '''
        self.tracked_dataset_names = set()
        self.mode = mode
        self.h5f = h5py.File(fname, mode)
        self.append_happened = False

    def __len__(self):
        if len(self.tracked_dataset_names) == 0:
            return 0
        else:
            for ds_name in self.tracked_dataset_names:
                return len(self.h5f[ds_name])


    def append_to_dataset(self, dataset_name, data):
        if len(data)==0:
            return

        try:
            self.h5f[dataset_name].resize(self.h5f[dataset_name].shape[0] + data.shape[0], axis=0)
            self.h5f[dataset_name][-data.shape[0]:] = data
        except Exception as e: # if there is no such dataset yet, create extendable datasets
            if len(data.shape)==1:
                ds_dim = [None]
            else:
                ds_dim = [None] + list(data.shape[1:])
            self.h5f.create_dataset(dataset_name, data=data,
                               compression="gzip",
                               compression_opts=9,
                               maxshape=ds_dim)


    def append(self, data, enforce_length = True):
        '''
        Append a new slice of data, assume that data.shape[1:] never changes
        :param data: a dict {dataset_name: new_data}, all chunks must have the same length
        :param enforce_length: Check that all the datasets are incremented equally,
        add the new ones to the list if not there yet
        :return:
        '''
        if type(data) != dict:
            return self.append({'data':data})

        if not enforce_length:
            # make sure we're not affecting controlled datasets
            assert(not any([ds_name in self.tracked_dataset_names for ds_name in data.keys()]))
            # and just append
            for ds_name, new_data in data.items():
                self.append_to_dataset(ds_name, new_data)
        else:
            # all new data chunks must have the same length for train/valid indices to work
            new_len = None
            for ds_name, new_data in data.items():
                if new_len is None:
                    new_len = len(new_data)
                else:
                    assert (new_len == len(new_data))

            if new_len == 0:
                return

            if len(self.tracked_dataset_names)==0 or not self.append_happened: # this is the first append since open
                # tracked_dataset_names could have a guessed value from train_valid_loaders
                self.tracked_dataset_names = set(data.keys())
                self.append_happened = True
            else:
                assert(set(data.keys()) == self.tracked_dataset_names)
                old_len = None
                for ds_name in self.tracked_dataset_names:
                    if old_len is None:
                        old_len = len(self.h5f[ds_name])
                    else:
                        assert(old_len == len(self.h5f[ds_name]))
            # now that all checks are done, let's append
            for ds_name, new_data in data.items():
                self.append_to_dataset(ds_name, new_data)

class SamplingWrapper:
    def __init__(self, h5wrapper, valid_frac=0.1, seq_len_name = 'seq_len'):
        self.valid_frac = valid_frac
        self.storage = h5wrapper
        self.seq_len_name = 'seq_len'
        self.idx = {True: None, False: None}
        self.sample_ind = None

    def augment_transient_indices(self):
        # Need to do this internally before every call as we assume
        # the underlying datasets may be growing in between.
        # Check for length of self.h5 data,
        # if necessary, create/update train, valid index arrays
        # if necessary, create/update
        ds_len = self.get_len(None,False) # length of underlying dataset
        my_len = self.get_len(True,False)+self.get_len(False,False)# length of my indices
        if ds_len == my_len:
            return
        else:
            new_len = ds_len - my_len

            # randomly assign new data indices to valid/train subsets
            is_valid = np.array([self.valid_frac >= np.random.uniform(size=new_len)])[0]
            all_ind = my_len + np.array(range(new_len))
            valid_ind = all_ind[is_valid]
            train_ind = all_ind[is_valid == False]
            for data_, valid in (valid_ind, True), (train_ind, False):
                if not len(data_):
                    continue
                else:
                    if self.idx[valid] is None:
                        self.idx[valid] = data_
                    else:
                        self.idx[valid] = np.concatenate([self.idx[valid], data_], axis=0)
            assert(ds_len == self.get_len(True,False)+self.get_len(False,False))

            # now initialize the extra sample indices
            # sample a number 0<=x<seq_len
            if self.seq_len_name in self.storage.tracked_dataset_names:
                if self.sample_ind is None:
                    seq_len = self.storage.h5f[self.seq_len_name]
                    self.sample_ind = np.floor(np.random.uniform(size=new_len) * seq_len * 0.9999).astype(int)
                else:
                    new_seq_len = self.storage.h5f[self.seq_len_name][len(self.sample_ind):]
                    new_sample_ind = np.floor(np.random.uniform(size=new_len)*new_seq_len*0.9999).astype(int)
                    self.sample_ind = np.concatenate([self.sample_ind, new_sample_ind], axis=0)
                assert(len(self.sample_ind) == ds_len)

    def get_item(self, ds_name, item, valid=None):
        '''
        Get a slice from a named dataset from self.storage
        :param ds_name: str (NOT list(str))
        :param item: int
        :param valid: if True/False, sample from the validation/training subset.
        If None, sample from the whole dataset
        :return:
        '''
        self.augment_transient_indices()
        if ds_name == 'sample_seq_ind':
            dataset = self.sample_ind
        else:
            dataset = self.storage.h5f[ds_name]

        if valid is None:
            return dataset[item]
        elif self.idx[valid] is not None:
            return dataset[self.idx[valid][item]]
        else:
            return None

    def get_len(self, valid, check_lengths = True):
        if check_lengths:
            self.augment_transient_indices()
        if valid is None:
            return self.storage.__len__()
        elif self.idx[valid] is not None:
            return len(self.idx[valid])
        else:
            return 0

    def check_dataset_names(self, dataset_name):
        check_list = [dataset_name] if type(dataset_name) == str else dataset_name
        check_list = [ds_name for ds_name in check_list if ds_name != 'sample_seq_ind']

        if len(self.storage.tracked_dataset_names):
            assert (all([ds_name in self.storage.tracked_dataset_names for ds_name in check_list]))
        else:
            # if we never appended to this dataset, just opened it, need to init
            for ds_name in check_list:
                try:
                    self.storage.h5f[ds_name]
                    self.storage.tracked_dataset_names.add(ds_name)
                except:
                    raise ValueError("no dataset called " + ds_name)
        return True # if we got this far

    def get_train_valid_loaders(self, batch_size, dataset_name='data'):
        # check that the required dataset(s) actually exist

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


class ChildHDF5Dataset(Dataset):
    def __init__(self, parent, dataset_name='data', valid=None):
        '''
        A Dataset facade for sampling from a, potentially live incrementing, hdf5 file or similar
        :param parent: The dataset wrapper providing the actual functionality
        :param dataset_name: str or list(str), what datasets to get from the source
        :param valid: If True or False, sample from validation or training subset, respectively
        if None, sample from the whole dataset
        '''
        self.parent = parent
        self.valid = valid
        self.dataset_name = dataset_name
        self.dataset_checked = False

    def __len__(self):
        if not self.dataset_checked:
            self.dataset_checked = self.parent.check_dataset_names(self.dataset_name)
        return self.parent.get_len(self.valid)

    def __getitem__(self, item):
        if item < self.__len__():
            # if we got this far, self.__len__() is >0 so we have indices
            if type(self.dataset_name) == str:
                return self.parent.get_item(self.dataset_name,
                                            item,
                                            self.valid
                                            )
            elif type(self.dataset_name) in (tuple, list):
                out = tuple(self.parent.get_item(dsname,
                                                 item,
                                                 self.valid) for dsname in self.dataset_name)
                return out
        else:
            raise ValueError("index exceeds dataset length")