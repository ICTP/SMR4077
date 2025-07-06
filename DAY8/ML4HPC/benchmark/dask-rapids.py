import dask
import cupy as cp
import cudf
import dask_cudf
from dask_cuda import LocalCUDACluster
from dask.distributed import Client, wait
import matplotlib.pyplot as plt
import time
import os

client = Client(scheduler_file=os.environ['SCHEDULER_FILE'])
print(client)

print("Creating synthetic distributed GPU data...")
n_rows = 100_000_000  # 100 million rows
n_parts = len(client.has_what())  # one partition per worker/GPU

# Create a Dask-cuDF DataFrame
ddf = dask_cudf.from_cudf(
    cudf.DataFrame({
        'x': cp.random.rand(n_rows),
        'y': cp.random.rand(n_rows),
        'label': cp.random.choice([10,100,1000], n_rows)
    }),
    npartitions=n_parts
)
ddf = ddf.persist()
wait(ddf)
print("DataFrame ready with", ddf.npartitions, "partitions")

print("Performing operations (mean, groupby)...")
start = time.time()
mean_x = ddf['x'].mean().compute()
grouped = ddf.groupby('label').y.mean().compute()
print("Mean of x:", mean_x)
print("Grouped y mean:\n", grouped)
print("Completed in", time.time() - start, "seconds")

print("Plotting a GPU-based scatter sample...")
sample = ddf.sample(frac=0.001).compute().to_pandas()

plt.scatter(sample['x'], sample['y'], alpha=0.3)
plt.title("Sampled GPU-parallel data")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

print("Applying CuPy UDF per partition")

def cupy_sigmoid(col):
    return 1 / (1 + cp.exp(-col))

def apply_sigmoid(df):
    df['sigmoid_x'] = cupy_sigmoid(df['x'].values)
    return df

sig_ddf = ddf.map_partitions(apply_sigmoid)
sig_ddf = sig_ddf.persist()
wait(sig_ddf)
print(sig_ddf.head())

# Done
print("All tasks complete!")

