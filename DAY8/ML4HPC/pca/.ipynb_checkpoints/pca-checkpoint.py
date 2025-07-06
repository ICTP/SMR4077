from dask_cuda import LocalCUDACluster
from distributed import Client
from dask.distributed import wait
from cuml.dask.decomposition import PCA
from cuml.dask.datasets import make_blobs
import numpy as np
import cupy as cp
import pandas as pd
import dask_cudf 
import os
import gc
import dask
import hvplot.dask
import holoviews as hv
import hvplot
import hvplot.dask
import hvplot.pandas
import dask.array as da
hv.extension('bokeh')

def main():
    # Connect to a cluster through a Dask client 
    os.environ["DASK_DISTRIBUTED__COMM__UCX__NVLINK"] = "True"
    os.environ["DASK_DISTRIBUTED__COMM__UCX__INFINIBAND"] = "True"
    os.environ["DASK_DISTRIBUTED__COMM__UCX__NET_DEVICES"] = "ib0"
    os.environ["DASK_DISTRIBUTED__COMM__UCX__CREATE_CUDA_CONTEXT"]="True"
    os.environ["DASK_DISTRIBUTED__COMM__UCX__CUDA_COPY"]="True"
    os.environ["DASK_DISTRIBUTED__COMM__UCX__TCP"]="True"
    os.environ["DASK_DISTRIBUTED__COMM__UCX__NVLINK"]="True"
    os.environ["DASK_DISTRIBUTED__COMM__UCX__INFINIBAND"]="True"
    os.environ["DASK_DISTRIBUTED__COMM__UCX__RDMACM"]="True"
    os.environ["UCX_MEMTYPE_REG_WHOLE_ALLOC_TYPES"]="cuda"
    os.environ["UCX_MEMTYPE_CACHE"]="n"
    dask.config.set({"dataframe.backend": "cudf"})
    dask.config.set({"array.backend": "cupy"})
    # print client info
    cluster = LocalCUDACluster(CUDA_VISIBLE_DEVICES=[0, 1, 2, 3],
                               n_workers=4,
                               threads_per_worker=8,
                               protocol="ucx",
                               interface="ib0",
                               enable_tcp_over_ucx=True,
                               enable_infiniband=True,
                               enable_nvlink=True,
                               enable_rdmacm=True,
                               rmm_pool_size="40GB",)

    client = Client(cluster)
    client.wait_for_workers(4)
    # print cluster and client info
    print("\ncluster:\n", cluster, flush=True)
    print("\nclient:\n",  client,  flush=True)
    # Get the number of workers
    n_workers = len(client.scheduler_info()["workers"].keys())
    assert n_workers == 4
    # Specify the dataset we are going to generate
    n_rows = 1000000
    n_cols = 12
    n_parts = 32
    # Get the blobs dataset X
    X, _ = make_blobs(n_samples=n_rows, n_features=n_cols, centers=1, n_parts=n_parts, cluster_std=0.01, random_state=10)
    wait(X)
    print("\nX:\n", X, flush=True)
    # Define a diagonal matrix to scale the dataset    
    sig = cp.array([4.0, 0.1, 0.1, 0.4, 0.5, 0.2, 0.3, 0.4, 2.0, 0.6, 0.2, 0.3])
    S = da.from_array(cp.diag(sig), asarray=False)
    wait(S)
    X = dask.array.matmul(X, S)
    wait(X)
    # Now the dataset is not anymore an isotropic Gaussian blob as we have scaled the features according to S
    # Define the distributed PCA model
    cumlModel = PCA(n_components = 2, whiten=False)
    # Compute the PCA by fitting over the dataset and returning the reduced dimension new dataset
    XT = cumlModel.fit_transform(X)
    XT_persisted = XT.persist()
    XT_persisted.compute_chunk_sizes() 
    print("\nXT_persisted:\n", XT_persisted, flush=True)
    #XT_persisted_computed = XT_persisted.compute()
    # Convert inbto a dataframe, plot and save into file
    df = dask.dataframe.from_dask_array(XT_persisted, columns=['x', 'y'])
    res=hv.Scatter(df.compute(),)
    res = res.opts(width=800, height=400)
    hv.save(res, 'res.html', backend='bokeh')
    #df_pandas= pd.DataFrame({'x': XT_persisted_computed[:, 0].get(), 'y': XT_persisted_computed[:, 1].get()}) 
    #res2=df_pandas.hvplot.scatter(x='x', y='y', height=400, width=400)
    #hv.save(res2, 'res2.html', backend='bokeh')
    client.close()

if __name__ == "__main__":
    main()

