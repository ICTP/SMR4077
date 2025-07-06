from distributed import Client
from dask.distributed import wait
from cuml.dask.datasets import make_blobs
from cuml.dask.neighbors import NearestNeighbors
import numpy as np
import cupy as cp
import os
import gc



# Generate distributed index and query
def generate_dist_dataset(client, n_samples, n_features, n_queries, n_centers, n_parts=None, n_workers=None):
    n_samples, n_features, n_queries = int(n_samples), int(n_features), int(n_queries)
    n_parts = int(n_parts) if n_parts else None
    # Select n_workers workers
    workers = list(client.has_what().keys())[:n_workers]
    # Generate index of n_parts partitions on n_workers workers
    index, _ = make_blobs(client=client, workers=workers, centers=n_centers, n_samples=n_samples, n_features=n_features, n_parts=n_parts)
    index = index.persist() # Start generation
    wait(index) # Wait for generation to finish
    # Generate single partition query on a worker
    query, _ = make_blobs(client=client, workers=workers, centers=n_centers, n_samples=n_queries, n_features=n_features, n_parts=1,)
    query = query.persist() # Start generation
    wait(query) # Wait for generation to finish
    return index, query


# Launch distributed KNN query
def cu_dist_query(client, d_index, d_query, n_neighbors):
    # Create cuML distributed KNN model
    model = NearestNeighbors(client=client, n_neighbors=n_neighbors, batch_size=16384)
    # batch_size is the max number of queries processed at one time (default 1024)
    # Fit model with index
    model.fit(d_index)
    # Start computation of search with query
    distances, indices = model.kneighbors(d_query)
    # Gather results of computation on client
    distances, indices = client.compute([distances, indices])
    wait([distances, indices])
    return distances, indices


# Make sure that deleted objects are released properly
def cleanup_memory(client):
    gc.collect() # Release memory on client
    client.run(gc.collect) # Release memory on cluster


def main():
    
    # Connect to a cluster through a Dask client
    client = Client(scheduler_file='out/scheduler.json')
    client.wait_for_workers(8)
    n_workers = len(client.scheduler_info()["workers"].keys())
    assert n_workers == 8
    # Specify the dataset (isotropic Gaussian blobs) in terms of the number of samples, features, queries, centers and number of partitions
    MAX_SAMPLES_PER_WORKER = 10000000
    n_samples = 20000000
    if n_samples > n_workers * MAX_SAMPLES_PER_WORKER:
        n_samples = n_workers * MAX_SAMPLES_PER_WORKER
    n_features = 256
    n_queries = 10000
    n_centers = 3
    n_parts = max(int(n_samples / 1000000), n_workers)
    print("\nn_parts:\n", n_parts, flush=True)
    # Generate the index and query dataset
    dist_index, dist_query = generate_dist_dataset(client, n_samples, n_features, n_queries, n_centers, n_parts=n_parts, n_workers=n_workers)
    print("\ndist_index:\n", dist_index, flush=True) 
    print("\ndist_query:\n", dist_query, flush=True) 
    # Specify the number of neighbors
    n_neighbors = 6
    # Fit the kNN model over the index and search with query
    distances, indices = cu_dist_query(client, dist_index, dist_query, n_neighbors)
    distances = distances.result()
    indices   = indices.result()
    print("\ndistances:\n", distances.shape, distances.dtype, flush=True)
    print("\n", distances, flush=True)
    print("\nindices:\n",   indices.shape,   indices.dtype,   flush=True)
    print("\n", indices, flush=True)
    # Clean the memory and close the client
    cleanup_memory(client)
    client.close()


if __name__ == "__main__":
    main()

