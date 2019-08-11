import tempfile
import uuid
from unittest import TestCase

import dill

from generative_playground.data_utils.shelve import Shelve


class DummyObject:
    def __init__(self, x):
        self.x = x

    def __eq__(self, other):
        return self.x == other.x


class ShelveTest(TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_dir_path = self.temp_dir.name
        self.db_path = '/{}/test.db'.format(self.temp_dir_path)
        self.table_name = 'kv_test'
        self.shelve = Shelve(self.db_path, self.table_name)
        self.table = self.shelve.db.table

    def tearDown(self):
        if self.shelve.is_open:
            self.shelve.close()
        self.temp_dir.cleanup()

    def add_dummy_object(self, x):
        key = uuid.uuid4()
        value = DummyObject(x)
        self.shelve[key] = value
        return key, value

    def get_all_pairs(self, shelve=None):
        if shelve is None:
            shelve = self.shelve
        rows = shelve.conn.execute(self.table.select()).fetchall()
        return {r['key']: dill.loads(r['value']) for r in rows}

    def test_add_to_cache(self):
        key, value = self.add_dummy_object(1)

        self.assertIn(key, self.shelve.cache)
        self.assertIs(self.shelve[key], value)

    def test_sync(self):
        key, value = self.add_dummy_object(1)

        self.assertIn(key, self.shelve.cache)
        self.assertIs(self.shelve[key], value)

        self.shelve.sync()

        self.assertNotIn(key, self.shelve.cache)
        self.assertIsNot(self.shelve[key], value)

        recovered = self.shelve[key]
        self.assertEqual(value, recovered)

    def test_update_value(self):
        key, value = self.add_dummy_object(1)
        self.shelve.sync()

        second = DummyObject(2)
        self.shelve[key] = second
        self.shelve.sync()

        recovered = self.shelve[key]
        self.assertEqual(second, recovered)

    def test_delete_item_in_cache(self):
        key, value = self.add_dummy_object(1)

        del self.shelve[key]

        self.assertNotIn(key, self.shelve.cache)

    def test_delete_item_in_store(self):
        key, value = self.add_dummy_object(1)
        self.shelve.sync()

        del self.shelve[key]

        self.assertNotIn(key, self.shelve.cache)

        with self.assertRaises(KeyError):
            self.shelve[key]
