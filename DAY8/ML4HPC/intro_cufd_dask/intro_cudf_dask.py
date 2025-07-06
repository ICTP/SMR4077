# Based on Introduction to Dask using cuDF DataFrames
# By Paul Hendricks
# https://github.com/rapidsai-community/notebooks-contrib/blob/main/getting_started_materials/intro_tutorials_and_guides/04_Introduction_to_Dask_using_cuDF_DataFrames.ipynb

from dask.distributed import Client, LocalCluster
from dask.distributed import wait
from dask.delayed import delayed
from dask_cuda import LocalCUDACluster
import cudf
import numpy as np
import dask_cudf 

# We'll define a function called load_data that will create a cudf.DataFrame with two columns, key and value. 
# The column key will be randomly filled with either a 0 or a 1, with 50% probability of either number being selected.
# The column value will be randomly filled with numbers sampled from a normal distribution.


def load_data(n_rows):
    df = cudf.DataFrame()
    random_state = np.random.RandomState(43210)
    df['key'] = random_state.binomial(n=1, p=0.5, size=(n_rows,))
    df['value'] = random_state.normal(size=(n_rows,))
    return df


# We'll also define a function head that takes a cudf.DataFrame and returns the first 5 rows.

def head(dataframe):
    return dataframe.head()


# We'll define a function called length that will take a cudf.DataFrame and return the first value of the shape attribute. 
# That is the number of rows for that particular dataframe.

def length(dataframe):
    return dataframe.shape[0]


# And we define a function groupby that takes a cudf.DataFrame, groups by the key column, and calculates the mean of the value column.

def groupby(dataframe):
    return dataframe.groupby('key')['value'].mean()



def main():
    # Let's start by creating a local cluster of workers and a client to interact with that cluster.
    # A local cluster with 4 workers
    n_workers = 4
    cluster = LocalCluster(n_workers=n_workers)
    client = Client(cluster)
    # show current Dask status
    print(client)
    # define the number of rows each dataframe will have
    n_rows = 125000000  # we'll use 125 million rows in each dataframe
    # create each dataframe using a delayed operation
    dfs = [delayed(load_data)(n_rows) for i in range(n_workers)]
    print("\ndfs:\n", dfs, flush=True)
    print("\nWe see the result of this operation is a list of Delayed objects!\n", flush=True)
    # It's important to note that these operations are "delayed" - nothing has been computed yet.
    # It means that our data has not yet been created!
    # We can apply the head function to each of our "delayed" dataframes.
    head_dfs = [delayed(head)(df) for df in dfs]
    print("\nhead_dfs:\n", head_dfs, flush=True)
    print("\nAs before, we see that the result is a list of Delayed objects!\n", flush=True)
    print("\nNote is that our key, or unique identifier for each operation, has changed!\n", flush=True)
    # You should see the name of the function head followed by a hash sign.
    # Again, nothing has been computed - let's compute the results and execute the workflow using the client.compute() method.
    # use the client to compute - this means create each dataframe and take the head
    futures = client.compute(head_dfs)
    wait(futures)  # this will give Dask time to execute the work before moving to any subsequently defined operations
    print("\nfutures:\n", futures, flush=True)
    print("\nWe see that our results are a list of futures.\n", flush=True)
    # Each object in this list tells us a bit information about itself:
    #   - the status (pending, error, finished),
    #   - the type of the object, 
    #   - and the key (unique identified).
    # We can use the client.gather method to collect the results of each of these futures.
    # collect the results
    results = client.gather(futures)
    print("\nresult (client.gather(futures)):\n", results)
    print("\nWe see that our results are a list of cuDF DataFrames, each having 2 columns and 5 rows.\n", flush=True) 
    # let's inspect the head of the first dataframe
    print("\nresults[0]:\n", results[0], flush=True)
    # Voila!
    # That was a pretty simple example.
    # Let's see how we can use this perform a more complex operation.
    # Like figuring how many total rows we have across all of our dataframes.
    # We'll define our operation on the dataframes we've created:
    lengths = [delayed(length)(df) for df in dfs]
    # And then use Python's built-in sum function to sum all of these lengths.
    total_number_of_rows = delayed(sum)(lengths)
    # At this point, total_number_of_rows hasn't been computed yet.
    # For each worker, we will first execute the load_data function to create each dataframe. 
    # Then the function length will be applied to each dataframe.
    # The results from these operations on each worker will then be combined into a single result via the sum function.
    # Let's now execute our workflow and compute a value for the total_number_of_rows variable.
    # use the client to compute the result and wait for it to finish
    future = client.compute(total_number_of_rows)
    wait(future)
    print("\nfuture (client.compute(total_number_of_rows)):\n", future, flush=True)
    print("\nWe see that our computation has finished - our result is of type int.\n", flush=True)
    # We can collect our result using the client.gather() method.
    # collect result
    result = client.gather(future)
    print("\nresult (client.gather(future)):\n", result, flush=True)
    # That's all there is to it! 
    # We can define more complex operations and workflows using cuDF DataFrames by using 
    # - delayed, 
    # - wait, 
    # - client.submit(), 
    # - client.gather()
    #
    # However, there can sometimes be a drawback from using this pattern. 
    # For example, consider a common operation such as a groupby.
    # We might want to group on certain keys and aggregate the values to compute a mean, variance, or even more complex aggregations.
    # Each dataframe is located on a different GPU.
    # We're not guaranteed that all of the keys necessary for that groupby operation are located on a single GPU. 
    # That is the keys may be scattered across multiple GPUs.
    # To make our problem even more concrete:
    # Let's consider the simple operation of grouping on our key column and calculating the mean of the value column.
    # To solve this problem, we'd have to sort the data and transfer keys and their associated values from one GPU to another.
    # Below, we'll show an example of this issue with the delayed pattern and motivate why one might consider using the dask_cudf API.
    # We'll apply the function groupby to each dataframe using the delayed operation.
    groupbys = [delayed(groupby)(df) for df in dfs]
    # We'll then execute that operation using the client to compute the result and wait for it to finish
    groupby_dfs = client.compute(groupbys)
    wait(groupby_dfs)
    print("\ngroupby_dfs:\n", groupby_dfs, flush=True)
    results = client.gather(groupby_dfs)
    print("\nresults:\n\n", results, flush=True)
    print("\nThat the following list of cuDF DataFrame:\n", flush=True)
    for i, result in enumerate(results):
        print('\ncuDF DataFrame:', i, flush=True)
        print(result, flush=True)
        print("\n", flush=True)
    #    
    print("\nThis isn't exactly what we wanted though\n", flush=True)
    print("\nIdeally, we'd get one dataframe where for each unique key (0 and 1), we get the mean of the value column\n", flush=True)
    # We can use the dask_cudf API to help up solve this problem. 
    # First we'll import the dask_cudf library
    # Then use the dask_cudf.from_delayed function to convert our list of delayed dataframes to a dask_cudf.core.DataFrame.
    # We'll use this object - distributed_df - along with the dask_cudf API to perform that "tricky" groupby operation.
    # create a distributed cuDF DataFrame using Dask
    distributed_df = dask_cudf.from_delayed(dfs)
    print("\ndistributed_df (dask_cudf.from_delayed(dfs)):\n", distributed_df, flush=True)
    print('\nType of distributed_df:\n', type(distributed_df))
    # The dask_cudf API closely mirrors the cuDF API.
    # We can use a groupby similar to how we would with cuDF - but this time, our operation is distributed across multiple GPUs!
    result = distributed_df.groupby('key')['value'].mean().compute()
    # Lastly, let's examine our result!
    print("\nresult:\n\n", result, flush=True)
    print("\n", flush=True)
    client.close()

if __name__ == "__main__":
    main()

