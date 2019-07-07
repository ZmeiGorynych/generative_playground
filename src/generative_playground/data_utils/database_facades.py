import os
import random
from typing import Dict, List

import dill
import torch
from sqlalchemy import create_engine
from torch.utils.data import DataLoader

from .experience_buffer_schema import create_exp_buffer


def unwind_batch(batch: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
    """
    Takes a batch of experiences and returns a list of single experiences
    """
    rewards = batch['reward']
    prev_states = batch['prev_state']
    states = batch['state']
    actions = batch['action']
    return [
        {
            'reward': r,
            'prev_state': p,
            'state': s,
            'action': a
        }
        for r, p, s, a in zip(rewards, prev_states, states, actions)
    ]


class DatabaseDeque:
    def __init__(self, db_path, max_len=float('inf')):
        self.max_len = max_len
        self.db_path = db_path
        self.db_url = "sqlite://{}".format(db_path)
        self.dbase = None
        self.table = None
        self.conn = None
        self.indices = []
        self.open_()

    def append(self, item, respect_max_len=True):
        item_pkl = dill.dumps(item)
        result = self.conn.execute(
            self.table.insert().values(data=item_pkl)
        )
        self.indices.extend(result.inserted_primary_key)
        if respect_max_len and len(self) > self.max_len:
            self.remove_items(1)

    def append_batch(self, batch):
        for exp in unwind_batch(batch):
            self.append(exp, respect_max_len=False)
        overflow = len(self) - self.max_len
        if overflow > 0:
            self.remove_items(overflow)

    def remove_items(self, num_items):
        del_ids = [self.indices.pop(0) for _ in range(num_items)]
        self.conn.execute(
            self.table.delete().where(
                self.table.c.id.in_(del_ids)
            )
        )

    def update_data(self, new_data):
        batch_size, num_steps = new_data['rewards'].shape
        for s in range(num_steps):
            for b in range(batch_size):
                old_state = new_data['env_outputs'][s][0]
                # exclude padding states
                if old_state[0][b] is None or len(old_state[0][b].nonterminal_ids()):
                    new_state = new_data['env_outputs'][s + 1][0]
                    reward = new_data['rewards'][b:b + 1, s]
                    action = new_data['actions'][s][b]
                    exp_tuple =(
                        slice(old_state, b),
                        action,
                        reward,
                        slice(new_state, b)
                    )
                    self.append(exp_tuple)

    def __getitem__(self, index):
        row = self.conn.execute(
            self.table.select().where(
                self.table.c.id == self.indices[index]
            )
        ).fetchone()
        return dill.loads(row['data'])

    def sample(self, num_samples, batch_size=1) -> DataLoader:
        ids = random.sample(self.indices, num_samples)
        rows = self.conn.execute(
            self.table.select().where(
                self.table.c.id.in_(ids)
            )
        ).fetchall()
        data = [dill.loads(r['data']) for r in rows]
        return DataLoader(data, batch_size=batch_size, shuffle=True)

    def __len__(self):
        return len(self.indices)

    def __enter__(self):
        self.open_()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def open_(self):
        self.dbase = create_engine(self.db_url)
        self.table = create_exp_buffer(self.dbase)
        self.conn = self.dbase.connect()

    def close(self):
        self.conn.close()
        if self.db_path:
            # Not in memory
            os.remove(self.db_path)
