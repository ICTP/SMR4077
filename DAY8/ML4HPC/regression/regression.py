from distributed import Client
from dask_cuda import LocalCUDACluster
from dask.distributed import wait
from cuml.dask.datasets.regression import make_regression
from cuml.dask.linear_model import LinearRegression
import numpy as np
import cupy as cp
import os
import gc


# Generate distributed regression dataset
def generate_dist_dataset(client, n_samples, n_features, n_informative, n_targets, n_parts, bias, noise=0.0):
    n_samples, n_features, n_informative, n_targets = int(n_samples), int(n_features), int(n_informative), int(n_targets)
    n_parts = int(n_parts) if n_parts else None
    X, y, coef = make_regression(client=client, n_samples=n_samples, n_features=n_features, n_informative=n_informative, 
                                 n_targets=n_targets, n_parts=n_parts, random_state=10, bias=bias, noise=noise, coef=True)
    return X, y, coef


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
    cluster = # ???
    client = # ???
    client.wait_for_workers(4)
    # print cluster and client info
    print("\ncluster:\n", cluster, flush=True)
    print("\nclient:\n",  client,  flush=True)
    # Get the number of workers
    n_workers = len(client.scheduler_info()["workers"].keys())
    assert n_workers == 4
    
    # Set up the number of samples
    MAX_SAMPLES_PER_WORKER = 80000000
    n_samples = 320000000
    if n_samples > n_workers * MAX_SAMPLES_PER_WORKER:
        n_samples = n_workers * MAX_SAMPLES_PER_WORKER
 
    # Specify further informations for the dataset we are going to generate
    n_features = 64
    n_informative = 16
    n_targets = 1
    n_parts = max(int(n_samples / 40000000), n_workers)
    bias = 1.0
    # Get the dataset (X, y), together with the true coefficient of the regression dataset (coef) 
    X, y, coef = generate_dist_dataset(client, n_samples, n_features, n_informative, n_targets, n_parts, bias)
    print("\nX:\n", X, flush=True)
    print("\ny:\n", y, flush=True)
    # Define a Linear Regression model, see https://docs.rapids.ai/api/cuml/stable/api/ and use the options fit_intercept=True, normalize=False
    lr= # TODO
    # Fit the model over the data
    # TODO
    # Get the learned coefficient and bias, look at the documentation https://docs.rapids.ai/api/cuml/stable/api/
    lrcoef= # TODO
    lrbias= # TODO

    print("\nThe original coefficients:\n", coef.compute(), flush=True)
    print("\nThe original bias:\n", bias, flush=True)
    print("\nThe learnt coefficients:\n", lrcoef, flush=True)
    print("\nThe learnt bias:\n", lrbias, flush=True)

    client.close()


if __name__ == "__main__":
    main()
