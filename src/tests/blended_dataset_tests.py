from unittest import TestCase
import torch
from torch.utils.data import DataLoader
from generative_playground.data_utils.blended_dataset import EvenlyBlendedDataset
from collections import deque
from generative_playground.molecules.data_utils.zinc_utils import get_smiles_from_database

class TestBlendedDataset(TestCase):
    def test_combined_loader_tensor_in_tensor_out_no_labels(self):
        zeros_loader = [0]*19
        ones_loader = [1]*10
        dataset = EvenlyBlendedDataset([zeros_loader, ones_loader], labels=False)
        loader = DataLoader(dataset, shuffle=True, batch_size=5)
        for out in loader:
            assert type(out) == torch.Tensor

    def test_combined_loader_tensor_in_tensor_out_labels(self):
        zeros_loader = [0] * 19
        ones_loader = [1] * 10
        dataset = EvenlyBlendedDataset([zeros_loader, ones_loader], labels=True)
        loader = DataLoader(dataset, shuffle=True, batch_size=5)
        for out in loader:
            assert type(out) == dict
            assert 'X' in out
            assert 'dataset_index' in out
            assert len(out['X']) == len(out['dataset_index'])
            for x, label in zip(out['X'],out['dataset_index']):
                assert x == label
    #
    def test_combined_loader_strings_in_tensor_out_labels(self):
        dataset = EvenlyBlendedDataset([['aaa']*19, ['bbb']*10], labels=True)
        loader = DataLoader(dataset, shuffle=True, batch_size=5)
        for out in loader:
            assert type(out) == dict
            assert 'X' in out
            assert 'dataset_index' in out
            for x, label in zip(out['X'],out['dataset_index']):
                assert x == 'aaa' if label == 0 else x == 'bbb'

    def test_zinc_loaders(self):
        history_size = 1000
        history_data = deque(['aaa','aaa','aaa'], maxlen=history_size)
        zinc_data = get_smiles_from_database(100)
        dataset = EvenlyBlendedDataset([history_data,zinc_data], labels=True)
        loader = DataLoader(dataset, shuffle=True, batch_size=10)
        for batch in loader:
            assert type(batch) == dict
            assert 'X' in batch
            assert 'dataset_index' in batch
            assert len(batch['X']) == len(batch['dataset_index'])
            for x, label in zip(batch['X'], batch['dataset_index']):
                if label == 1:
                    assert x != 'aaa'
                elif label == 0:
                    assert x == 'aaa'
                else:
                    raise ValueError("Unknown label")
