from sqlalchemy import Column, Index, Integer, MetaData, BLOB, Table


def create_exp_buffer(engine) -> Table:
    meta = MetaData()

    exp_buffer = Table(
        'exp_buffer', meta,
        Column('id', Integer, primary_key=True),
        Column('data', BLOB)
    )
    Index('exp_buffer_id_idx', exp_buffer.c.id)
    meta.create_all(engine)
    return exp_buffer
