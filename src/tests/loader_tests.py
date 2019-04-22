from unittest import TestCase
import torch
from torch.utils.data import DataLoader
from generative_playground.data_utils.mixed_loader import CombinedLoader
from collections import deque
from generative_playground.molecules.data_utils.zinc_utils import get_zinc_smiles

class TestStart(TestCase):
    def test_combined_loader_tensor_in_tensor_out_no_labels(self):
        zeros_loader = DataLoader([0]*19, shuffle=True, batch_size=5)
        ones_loader = DataLoader([1]*10, shuffle=True, batch_size=5)
        loader = CombinedLoader([zeros_loader, ones_loader], num_batches=1)
        for out in loader:
            assert type(out) == torch.Tensor

    def test_combined_loader_tensor_in_tensor_out_labels(self):
        zeros_loader = DataLoader([0]*19, shuffle=True, batch_size=5)
        ones_loader = DataLoader([1]*10, shuffle=True, batch_size=5)
        loader = CombinedLoader([zeros_loader, ones_loader], num_batches=1, labels=[0,1])
        for out in loader:
            assert type(out) == dict
            assert 'X' in out
            assert 'labels' in out
            assert len(out['X']) == len(out['labels'])
            for x, label in zip(out['X'],out['labels']):
                assert x == label

    def test_combined_loader_strings_in_tensor_out_int_labels(self):
        zeros_loader = DataLoader(['aaa']*19, shuffle=True, batch_size=5)
        ones_loader = DataLoader(['bbb']*10, shuffle=True, batch_size=5)
        loader = CombinedLoader([zeros_loader, ones_loader], num_batches=1, labels=[0,1])
        for out in loader:
            assert type(out) == dict
            assert 'X' in out
            assert 'labels' in out
            for x, label in zip(out['X'],out['labels']):
                assert x == 'aaa' if label == 0 else x == 'bbb'

    def test_combined_loader_strings_in_tensor_out_string_labels(self):
        zeros_loader = DataLoader(['aaa']*19, shuffle=True, batch_size=5)
        ones_loader = DataLoader(['bbb']*10, shuffle=True, batch_size=5)
        loader = CombinedLoader([zeros_loader, ones_loader], num_batches=5, labels=['a', 'b'])
        for out in loader:
            assert type(out) == dict
            assert 'X' in out
            assert 'labels' in out
            assert len(out['X']) == len(out['labels'])
            for x, label in zip(out['X'],out['labels']):
                assert x == 'aaa' if label == 'a' else x == 'bbb'

    def test_zinc_loaders(self):
        history_size = 1000
        history_data = deque(['aaa','aaa','aaa'], maxlen=history_size)
        history_loader = DataLoader(history_data, shuffle=True, batch_size=5)

        zinc_data = get_zinc_smiles(100)
        zinc_loader = DataLoader(zinc_data, shuffle=True, batch_size=5)

        loader = CombinedLoader([history_loader, zinc_loader], num_batches=10, labels=[0,1])
        for batch in loader:
            assert type(batch) == dict
            assert 'X' in batch
            assert 'labels' in batch
            assert len(batch['X']) == len(batch['labels'])
            for x, label in zip(batch['X'], batch['labels']):
                if label == 1:
                    assert x != 'aaa'
                elif label == 0:
                    assert x == 'aaa'
                else:
                    raise ValueError("Unknown label")
