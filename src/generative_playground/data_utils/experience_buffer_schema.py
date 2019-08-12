from sqlalchemy import BLOB, Column, Index, MetaData, String, Table
from sqlalchemy.schema import UniqueConstraint


def create_kv_store(engine, name: str) -> Table:
    meta = MetaData()

    kv_store = Table(
        name,
        meta,
        Column('key', String, primary_key=True),
        Column('value', BLOB),
        UniqueConstraint('key', sqlite_on_conflict='REPLACE')
    )
    Index('{}_id_idx'.format(name), kv_store.c.key)
    meta.create_all(engine)
    return kv_store
