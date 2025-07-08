from cuml.dask.cluster import KMeans
from dask_cuda import LocalCUDACluster
from dask.distributed import Client, wait
import dask.array as da
import time
import os 
import pickle
from cuml.dask.datasets import make_blobs

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
    current_dir = os.environ.get('SLURM_SUBMIT_DIR')
    # Set up a LocalCUDACluster with 4 workers supposed to run on 4 GPUs in a single node
    # Experiment with the flags, for example enable ucx. What happens of we do not enable ucx?
    cluster = LocalCUDACluster(CUDA_VISIBLE_DEVICES=[0, 1, 2, 3],
                               n_workers=4,
                               threads_per_worker=8,
                               protocol="ucx",
                               interface="ib0",
                               enable_tcp_over_ucx=True,
                               enable_infiniband=True,
                               enable_nvlink=True,
                               enable_rdmacm=True,
                               rmm_pool_size="50GB",)

    client = Client(cluster)
    client.wait_for_workers(4)
    # print cluster and client info
    print("\ncluster:\n", cluster, flush=True)
    print("\nclient:\n",  client,  flush=True)
    # Set up the number of samples
    n_samples = 200000000
    # Specify further informations for the dataset we are going to generate
    n_features = 6
    start_random_state = 170
    n_parts = 20
    n_clusters = 8
    # Get the dataset (X, y), together with the true centers of the blobs dataset
    X, _, centers = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, cluster_std=0.4, 
                               random_state=start_random_state, n_parts=n_parts, return_centers=True)
    print("\nX:\n", X, flush=True)
    # Define the KMeans model and fit over the data
    # Use the KMeans from cuml.dask.cluster with n_clusters defined above 
    kmeans_model = # TODO
    # Fit kmeans_model over the blobs data
    # TODO
    # Compute the score over the dataset, computes the inertia score for the trained KMeans centroids.
    score = # TODO
    print(f"\nScore: {score}\n", flush=True)
    # The distributed estimator wrappers inside of the cuml.dask are not intended to be pickled directly. 
    # The Dask cuML estimators provide a function get_combined_model(), which returns the trained single-GPU model for pickling. 
    single_gpu_model = #TODO
    # Save the model to file
    pickle.dump(single_gpu_model,  open(current_dir + "/kmeans_model.pkl", "wb"))
    # Load the model from file
    single_gpu_model = pickle.load(open(current_dir + "/kmeans_model.pkl", "rb"))
    # Check the centers
    print("\nsingle_gpu_model.cluster_centers_:\n",  single_gpu_model.cluster_centers_, flush=True)
    print("\nTrue centers:\n", centers, flush=True)
    print("Closing...", flush=True)
    client.close()

if __name__ == '__main__':
    main()
