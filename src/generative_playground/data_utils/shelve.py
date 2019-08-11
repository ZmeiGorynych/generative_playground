import collections.abc
from io import BytesIO

import dill

from .database_facades import DBKVStore


__all__ = ["Shelve", "open"]


class _ClosedDict(collections.abc.MutableMapping):
    'Marker for a closed dict.  Access attempts raise a ValueError.'

    def closed(self, *args):
        raise ValueError('invalid operation on closed shelf')
    __iter__ = __len__ = __getitem__ = __setitem__ = __delitem__ = keys = closed

    def __repr__(self):
        return '<Closed Dictionary>'

    def write_many(self, *args, **kwargs):
        raise ValueError('invalid operation on closed shelf')


class Shelve(collections.abc.MutableMapping):
    def __init__(self, db_url, table_name):
        self.db = DBKVStore(db_url, table_name)
        self.cache = {}
        self.is_open = True

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.db)

    def __contains__(self, key):
        raise NotImplementedError

    def get(self, key, default=None):
        if key in self.db:
            return self[key]
        return default

    def __getitem__(self, key):
        try:
            value = self.cache[key]
        except KeyError:
            f = BytesIO(self.db[str(key)])
            value = dill.load(f)
            self.cache[key] = value
        return value

    def __setitem__(self, key, value):
        self.cache[key] = value

    def __delitem__(self, key):
        del self.db[str(key)]
        try:
            del self.cache[key]
        except KeyError:
            pass

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def close(self):
        if self.db is None or not self.is_open:
            return
        try:
            self.sync()
            self.db.close()
        finally:
            # Catch errors that may happen when close is called from __del__
            # because CPython is in interpreter shutdown.
            try:
                self.db = _ClosedDict()
            except Exception:
                self.db = None
            self.is_open = False

    def __del__(self):
        self.close()

    def sync(self):
        if self.db is None:
            raise ValueError('DB is closed!')
        if len(self.cache) > 0:
            self.db.write_many(
                {str(k): dill.dumps(v) for k, v in self.cache.items()}
            )
        self.cache = {}
