import palletjack as pj
import pyarrow.parquet as pq
import polars as pl
import numpy as np
import time
import concurrent.futures
import os
import pickle 

rows = 100
columns = 1000
chunk_size = 1 # A row group per
workers = 8
work_items = 200

path = "my.parquet"
index_path = path + '.index'

def worker_indexed():
    
    for r in range(0, rows):
        metadata = pj.read_row_group_metadata(index_path, r)
        pr = pq.ParquetReader()
        pr.open(path, metadata=metadata)
        res_data = pr.read_row_groups([0], column_indices=[0,1,2], use_threads=False)
    
def worker_pickle():

    pr = pq.ParquetReader()
    pr.open(path)
    metadata = pr.metadata

    for r in range(0, rows):
        m = pickle.loads(pickle.dumps(metadata))
        {}
    
def worker_indexed_metadata():

    for r in range(0, rows):
        metadata = pj.read_row_group_metadata(index_path, r)

def worker_org_metadata():
    
    for r in range(0, rows):
        pr = pq.ParquetReader()
        pr.open(path)
        m = pr.metadata
        {}

def worker():
    
    for r in range(0, rows):
        pr = pq.ParquetReader()
        pr.open(path)        
        res_data = pr.read_row_groups([0], column_indices=[0,1,2], use_threads=False)

if not os.path.isfile(path):
    
    table = pl.DataFrame(
        data=np.random.randn(rows, columns),
        schema=[f"c{i}" for i in range(columns)]).to_arrow()

    print ("writing table begin")
    pq.write_table(table, path, row_group_size=chunk_size, use_dictionary=False, write_statistics=False, store_schema=False)
    print ("writing table end")

    # Reading using the indexed metadata
    print ("generate_metadata_index begin")
    pj.generate_metadata_index(path, index_path)
    print ("generate_metadata_index end")

t = time.time()
print ("Starting threads")
pool = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
for i in range(0, work_items):
    pool.submit(worker_indexed)
 
pool.shutdown(wait=True)
dt = time.time() - t
print (f"Done {dt}s")