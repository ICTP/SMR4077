Adapted from https://medium.com/rapids-ai/scaling-knn-to-new-heights-using-rapids-cuml-and-dask-63410983acfe

In `knn.py` we show how to implement a distributed k-Nearest Neighbors (kNN) using cuml and dask, a machine learning method that can be used to build classification, regression models, or search algorithm.

In kNN, we want to predict the outcome of a particular observation in a dataset based on similar observations.

Such similarity is defined in the terms of a distance metric.

That is, for each new observation, we finds the k most similar (nearest neighbors according to the metric) observations with known outcomes.

In the following we will focus on the search, where observations (each consisting of a number of features) are gathered into a data structure known as the index.

New, unlabelled observations upon which the search is performed are called the query. 

The complexity of the kNN search is dependent upon the size of the index, query, and number of features.

Parallelization can increases the performance of kNN, which depends on the partitioning of index and query distributed arrays at the start of the algorithm and on the performance of the underlying networking interconnect used.

For a multi-node, multi-GPU (MNMG) kNN algorithm, we need the following ingredients:

- a scheduler that organizes the execution of task graphs on several workers

- a client that initiates the loading of the data (which will be then partitioned across workers) and triggers the distributed operations to perform the search

- multiple workers, each attached to a single-GPU, running the same set of instructions on their local data partitions possibly directly interacting with each other for communications or collective operations

The MNMG kNN search requires the global data to be partitioned in local arrays, distributed among the workers, forming the following distributed arrays or dataframes:

- dist_index: contains reference observations, size (n_samples x m_features)

- dist_query: contains observations for which to find the nearest neighbors, size (n_queries x m_features)

At the end of distributed search, the results that is returned are following arrays

- indices: follows the distribution of the query, will contain the indices of the nearest neighbors in the index for each of the query points, size (n_queries x n_neighbors)

- distances: follows the distribution of the query, will contain the distances of the nearest neighbors toward their respective query points, size (n_queries x n_neighbors)

Suppose we have an index dataset consisting of $n$ samples (rows) each consisting of $m$ features, for example, isotropic Gaussian blobs with n=20000000 samples and m=256 features where points are distributed around 3 centers divided into 20 parts.

dist_index:
 dask.array<concatenate, shape=(20000000, 256), dtype=float32, chunksize=(1000000, 256), chunktype=cupy.ndarray>


Similarly, for the query dataset we consider isotropic Gaussian blobs with n=10000 samples and m=256 features where points are distributed around 3 centers in one single part. 


dist_query:
 dask.array<from-value, shape=(10000, 256), dtype=float32, chunksize=(10000, 256), chunktype=cupy.ndarray>


Then, we focus on 6 nearest neighbors and so we instantiate cuml.dask.neighbors.NearestNeighbors:

model = NearestNeighbors(client=client, n_neighbors=n_neighbors, batch_size=16384)

Here, batch_size is the max number of queries processed at once (the default value is 1024).
This parameter can greatly affect the throughput of the algorithm. 

After fitting the model over the index dataset, we compute the neighbors:

distances, indices = model.kneighbors(dist_query)  

The results are two arrays.

distances:
 (10000, 6) float32

 [[124.787735 124.979935 125.07282  125.08751  125.08788  125.09493 ]
 [126.18769  126.20736  126.526665 126.58987  126.59528  126.6351  ]
 [125.8121   126.08045  126.13347  126.19919  126.24081  126.26461 ]
 ...
 [126.08376  126.08478  126.096634 126.20287  126.23112  126.23948 ]
 [118.585075 118.7907   118.99315  119.001205 119.01841  119.0264  ]
 [118.08345  118.1331   118.16193  118.32711  118.35559  118.409134]]


indices:
 (10000, 6) int64

 [[ 5680999  1688705 13046560 12946493  1786277 11016867]
 [ 9878134 10102114  9158079  4708533  6945876   397592]
 [12025044  7822886  8006246 12924129  5900684   294882]
 ...
 [12025044  2178741  8006246 12924129  7822886 13854088]
 [11668199 16434949 19715018  3757641 13892210  9045616]
 [16434949  3757641 11668199  7731959 11878459  9045616]]


As noted above queries are processed sequentially as a series of batches. 
For each batch of queries:

- Queries broadcasting: The unique owner of the currently processed batch of queries broadcasts it to all the other workers.

- Local kNN: All of the workers having at least one index partition run a local kNN. The local kNN searches for the nearest neighbors of the currently processed batch of queries in its locally stored index partitions. The produced indices and distances correspond to the local nearest neighbors for these query points.

- Local kNN results gathering: The workers that performed the previous step send the result of their local kNN back. They then start processing the next batch of queries. The owner of the currently processed batch of queries collects all of this data.

- Reduce: The owner of the currently processed batch then performs a reduce operation on the data it received. It consists in merging the results of the local kNN searches. This produces a set of global nearest neighbors indices for each of the queries along with the corresponding distances between query and newly found nearest neighbors.


At the end of distributed operations, once all of the queries have been processed, the user has a first distributed array with the set indices of the nearest neighbors for each query and a second one with the distances between each of the query points and their nearest neighbors.

