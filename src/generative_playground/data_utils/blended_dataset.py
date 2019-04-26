from torch.utils.data.dataset import Dataset
from collections import OrderedDict
from math import floor

class EvenlyBlendedDataset(Dataset):
    '''
    Creates a balanced dataset from a list of datasets, oversampling if necessary
    '''
    def __init__(self, dataset_list, labels=False):
        self.datasets = dataset_list
        self.max_len = max([len(d) for d in dataset_list])
        self.labels = labels

    def __len__(self):
        return self.max_len*len(self.datasets)

    def __getitem__(self, item):
        ds_index = floor(item/self.max_len)
        this_len = len(self.datasets[ds_index])
        pre_item_index = item % self.max_len
        item_index = pre_item_index % this_len
        pre_item = self.datasets[ds_index][item_index]
        if self.labels:
            if type(pre_item) in (dict, OrderedDict):
                if 'dataset_index' not in pre_item:
                    pre_item['dataset_index'] = ds_index
                    return pre_item
                else:
                    raise ValueError("Key 'dataset_index' is already used")
            else:
                pre_item = {'X': pre_item, 'dataset_index': ds_index}
        return pre_item