import torch
from generative_playground.utils.gpu_utils import to_gpu
import numpy as np
from collections import OrderedDict
from frozendict import frozendict


class MixedLoader:
    def __init__(self, main_loader, valid_ds, invalid_ds):
        self.main_loader = main_loader
        self.valid_ds = valid_ds
        self.invalid_ds = invalid_ds
        self.batch_size = main_loader.batch_size

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        def gen():
            iter1 = iter(self.main_loader)
            iter2 = iter(self.valid_ds)
            iter3 = iter(self.invalid_ds)
            while True:
                # make sure we iterate fully over the first dataset, others will likely be shorter
                x1 = next(iter1).float()
                try:
                    x2 = next(iter2).float()
                except StopIteration:
                    iter2 = iter(self.valid_ds)
                    x2 = next(iter2).float()
                try:
                    x3 = next(iter3).float()
                except StopIteration:
                    iter3 = iter(self.valid_ds)
                    x3 = next(iter3).float()

                x = to_gpu(torch.cat([x1,x2,x3], dim=0))
                y = to_gpu(torch.zeros([len(x),1]))
                y[:(len(x1)+len(x2))]=1
                yield x,y

        return gen()

class CombinedLoader:
    def __init__(self, ds_list, num_batches=100, labels=None):
        '''
        Glues together output from a list of loaders: goal is to produce a balanced dataset from the sources
        :param ds_list: A list of loaders. Each batch is either a list/tuple of whatever, a torch.Tensor, or a dict of these
        :param num_batches: Number of batches to return
        '''
        assert labels is None or len(labels) == len(ds_list), "Number of labels not equal to number of datasets"
        self.ds_list = ds_list
        self.labels = labels
        try: # TODO: this is legacy code, do we even need it?
            self.batch_size = self.ds_list[0].batch_size
        except:
            pass
        self.num_batches = num_batches

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        def gen():
            iters = [ds.__iter__() for ds in self.ds_list]
            for _ in range(self.num_batches):
                x_list = []
                labels = []
                for i, iter in enumerate(iters):
                    try:
                        x_list.append(next(iter))
                    except StopIteration:
                        iters[i] = self.ds_list[i].__iter__()
                        x_list.append(next(iters[i]))
                    if self.labels is not None:
                        try:
                            float(self.labels[i]) # if the labels are numeric
                            labels.append(self.labels[i]*torch.ones(batch_size(x_list[-1]), dtype=torch.long))
                        except ValueError:
                            labels.append([self.labels[i]] * batch_size(x_list[-1]))
                # TODO: make sure this supports dicts as well; refactor to use pytorch's collation function
                x = concat(x_list)
                if len(labels):
                    all_labels = concat(labels)
                    if type(x) in (dict, OrderedDict):
                        if 'labels' in x:
                            raise ValueError("key 'labels' is already used")
                        x['labels'] = all_labels
                        yield x
                    else: #create the dict
                        yield {'X': x, 'labels': all_labels}
                else:
                    yield x

        return gen()

def batch_size(x):
    if type(x) in (list, tuple):
        elem = x[0]
    elif type(x) in (dict, OrderedDict, frozendict):
        for x_ in x.values():
            elem = x_
            break
    elif type(x) == torch.Tensor:
        return x.size(0)

    if type(elem) == torch.Tensor:
        return elem.size(0)
    elif type(elem) in (list, tuple):
        return len(elem)
    else:
        return len(x)



def concat(x):
    if type(x) in (list, tuple):
        if type(x[0]) == torch.Tensor:
            return to_gpu(torch.cat(x, dim=0))
        elif type(x[0]) in (list, tuple):
            out = []
            for i in x:
                out += i
            return out
            #return [concat(elem) for elem in zip(*x)]
        elif type(x[0]) in (dict, OrderedDict):
            raise NotImplementedError("Can't handle loaders returning dicts yet")
        else:
            return x
    else:
        raise ValueError("Can only concatenate lists and tuples as yet")