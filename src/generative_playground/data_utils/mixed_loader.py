import torch
from generative_playground.gpu_utils import FloatTensor, to_gpu
class MixedLoader:
    def __init__(self, main_loader, valid_ds, invalid_ds):
        self.main_loader = main_loader
        self.valid_ds = valid_ds
        self.invalid_ds = invalid_ds
        self.batch_size = main_loader.batch_size


    def __len__(self):
        return self.num_batches# Wlen(self.main_loader)

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
    def __init__(self, ds_list, num_batches=100):
        '''
        Glues together output from a list of loaders: goal is to produce a balanced dataset from the 2 sources
        :param ds1:
        :param ds2:
        '''
        self.ds_list = ds_list
        self.batch_size = self.ds_list[0].batch_size
        self.num_batches = num_batches

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        def gen():
            iters = [iter(ds) for ds in self.ds_list]
            x_list = [None for _ in iters]
            for _ in range(self.num_batches):
                for i in range(len(iters)):
                    try:
                        x_list[i] = next(iters[i])
                    except StopIteration:
                        iters[i] = iter(self.ds_list[i])
                        x_list[i] = next(iters[i])

                if type(x_list[0]) == tuple or type(x_list[0]) == list:
                    x = tuple(to_gpu(torch.cat(elem, dim=0)) for elem in zip(*x_list))
                else:
                    x = to_gpu(torch.cat(x_list, dim=0))

                yield (x[0], x[1:])

        return gen()