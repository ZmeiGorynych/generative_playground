import copy
import os
import tempfile
from unittest import TestCase

import dill
import numpy as np
import torch
from torch.utils.data import DataLoader

from generative_playground.data_utils.database_facades import (
    DatabaseDeque, unwind_batch
)


def make_random_experience():
    return {
        'reward': torch.tensor(np.random.rand(1)),
        'prev_state': torch.tensor(np.random.rand(100)),
        'state': torch.tensor(np.random.rand(100)),
        'action': torch.tensor(np.random.randint(1, 200))
    }


class ExperienceBufferTest(TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_dir_path = self.temp_dir.name
        self.db_path = '/{}/test.db'.format(self.temp_dir_path)
        self.buffer_ = DatabaseDeque(self.db_path)
        self.table = self.buffer_.table
        self.experience = make_random_experience()

    def tearDown(self):
        self.buffer_.close()
        self.temp_dir.cleanup()

    def compare_experiences(self, exp1, exp2):
        for k, v in exp1.items():
            with self.subTest(key=k):
                self.assertTrue(torch.equal(v, exp2[k]))

    def append_random_experiences(self, num_exps, buffer_=None):
        if buffer_ is None:
            buffer_ = self.buffer_
        experiences = [make_random_experience() for _ in range(num_exps)]
        for exp in experiences:
            buffer_.append(exp)
        return experiences

    def append_random_experience_batch(self, batch_size, buffer_=None):
        if buffer_ is None:
            buffer_ = self.buffer_
        experiences = [make_random_experience() for _ in range(batch_size)]
        loader = DataLoader(experiences, batch_size=batch_size)
        for batch in loader:
            buffer_.append_batch(batch)
        return experiences

    def get_all_experiences(self, buffer_=None):
        if buffer_ is None:
            buffer_ = self.buffer_
        rows = buffer_.conn.execute(self.table.select()).fetchall()
        return [dill.loads(r['data']) for r in rows]

    def test_append(self):
        self.buffer_.append(self.experience)

        self.assertEqual(len(self.buffer_.indices), 1)
        self.assertIsInstance(self.buffer_.indices[0], int)

        rows = self.buffer_.conn.execute(self.table.select()).fetchall()
        self.assertEqual(len(rows), 1)

        res = dill.loads(rows[0]['data'])
        self.compare_experiences(res, self.experience)

    def test_append_max_len(self):
        path = '/{}/test2.db'.format(self.temp_dir_path)
        with DatabaseDeque(path, max_len=5) as window_buffer:
            self.append_random_experiences(5, buffer_=window_buffer)
            early_indices = copy.copy(window_buffer.indices)

            self.append_random_experiences(5, buffer_=window_buffer)
            latter_indices = copy.copy(window_buffer.indices)

            self.assertEqual(len(early_indices), 5)
            self.assertEqual(len(latter_indices), 5)
            self.assertTrue(
                set(early_indices).isdisjoint(set(latter_indices))
            )

    def test_append_batch(self):
        batch_size = 5
        experiences = self.append_random_experience_batch(batch_size)

        self.assertEqual(len(self.buffer_.indices), batch_size)
        self.assertIsInstance(self.buffer_.indices[0], int)

        rows = self.get_all_experiences()
        self.assertEqual(len(rows), 5)

        for i, res in enumerate(rows):
            self.compare_experiences(res, experiences[i])

    def test_append_batch_max_len(self):
        path = '/{}/test2.db'.format(self.temp_dir_path)
        with DatabaseDeque(path, max_len=5) as window_buffer:
            self.append_random_experience_batch(5, buffer_=window_buffer)
            early_indices = copy.copy(window_buffer.indices)

            self.append_random_experience_batch(5, buffer_=window_buffer)
            latter_indices = copy.copy(window_buffer.indices)

            self.assertEqual(len(early_indices), 5)
            self.assertEqual(len(latter_indices), 5)
            self.assertTrue(
                set(early_indices).isdisjoint(set(latter_indices))
            )

    def test_getitem(self):
        experiences = self.append_random_experiences(10)

        for i, exp in enumerate(experiences):
            res = self.buffer_[i]
            self.compare_experiences(res, experiences[i])

    def test_sample(self):
        experiences = self.append_random_experiences(500)

        num_samples = 100
        batch_size = 10
        sampled = self.buffer_.sample(num_samples, 10)

        self.assertEqual(len(sampled), num_samples // batch_size)

        # Test each appear once in the dataset
        counts = [0] * num_samples
        for i, batch in enumerate(sampled):
            exps = unwind_batch(batch)
            for j, sample in enumerate(exps):
                for exp in experiences:
                    try:
                        self.compare_experiences(sample, exp)
                    except AssertionError:
                        pass
                    else:
                        counts[i * batch_size + j] += 1

        self.assertTrue(all(count == 1 for count in counts))


class ExperienceBufferCloseTest(TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_dir_path = self.temp_dir.name
        self.db_path = '/{}/test.db'.format(self.temp_dir_path)
        self.buffer_ = DatabaseDeque(self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_open_close(self):
        self.assertTrue(os.path.exists(self.db_path))

        self.buffer_.close()

        self.assertFalse(os.path.exists(self.db_path))
