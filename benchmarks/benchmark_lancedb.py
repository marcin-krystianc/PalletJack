import time
import numpy as np
import palletjack as pj

import lance.fragment
import pyarrow as pa
import pyarrow.parquet as pq

from lance.file import LanceFileReader, LanceFileWriter

row_groups = 100
columns = 20000
chunk_size = 100
rows = row_groups * chunk_size
work_items = 1

parquet_path = "/tmp/my.parquet"
lancefile_path = "/tmp/my.lance"
lancedb_path = "/tmp/lancedb"
index_path = parquet_path + '.index'

def get_table():
    data = np.random.rand(rows, columns)
    pa_arrays = [pa.array(data[:, i]) for i in range(columns)]
    column_names = [f"column_{i}" for i in range(columns)]
    return pa.Table.from_arrays(pa_arrays, names=column_names)

def worker_arrow():
    for rg in range(0, row_groups):
        pr = pq.ParquetReader()
        pr.open(parquet_path)
        data = pr.read_row_groups([rg], use_threads=False)
        data = data

def worker_lance():
    for rg in range(0, row_groups):
        lr = LanceFileReader(lancefile_path)
        data = lr.read_range(rg * chunk_size, chunk_size).to_table()
        data = data
        
def worker_palletjack():    
    for rg in range(0, row_groups):
        metadata = pj.read_metadata(index_path, row_groups = [rg])
        pr = pq.ParquetReader()
        pr.open(parquet_path, metadata=metadata)
        pr.read_row_groups([0], use_threads=False)
        
def genrate_data(table, store_schema):
    t = time.time()
    print(
        f"writing parquet file, columns={columns}, row_groups={row_groups}, rows={rows}"
    )
    pq.write_table(
        table,
        parquet_path,
        row_group_size=chunk_size,
        use_dictionary=False,
        write_statistics=False,
        store_schema=store_schema,
        compression=None,
    )
    dt = time.time() - t
    print(f"finished writing parquet file in {dt:.2f} seconds")

    with LanceFileWriter(lancefile_path, table.schema) as writer:
        writer.write_batch(table)
    
    pj.generate_metadata_index(parquet_path, index_path)


def measure_reading(worker):
    t = time.time()
    for i in range(0, work_items):
        worker()

    return time.time() - t


table = get_table()
genrate_data(table, False)

print( f"Reading all row groups using arrow (single-threaded) {measure_reading(worker_arrow):.3f} seconds")
print(f"Reading all row groups using lance v2 {measure_reading(worker_lance):.3f} seconds")
print(f"Reading all row groups using PalletJack {measure_reading(worker_palletjack):.3f} seconds")