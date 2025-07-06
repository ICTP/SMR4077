# From Introduction to Dask By Paul Hendricks
# https://github.com/rapidsai-community/notebooks-contrib/blob/main/getting_started_materials/intro_tutorials_and_guides/03_Introduction_to_Dask.ipynb 

import dask
from dask.distributed import Client, LocalCluster
from dask import delayed
import time 
from dask.distributed import wait

# Introduction to Dask

# Dask is a library the allows for parallelized computing. 
# It allows one to compose complex workflows using large data structures like those found in NumPy, Pandas, and cuDF. 

# Client/Workers

# Dask operates by creating a cluster composed of a "client" and multiple "workers". 
# The client is responsible for scheduling work.
# The workers are responsible for actually executing that work.

# Typically, we set the number of workers to be equal to the number of computing resources we have available to us.
# For CPU based workflows, this might be the number of cores or threads on that particlular machine. 
# We might set n_workers = 8 if we have 8 CPU cores or threads on our machine that can each operate in parallel. 
# This allows us to take advantage of all of our computing resources and enjoy the most benefits from parallelization.

# On a system more GPUs, we usually set the number of workers equal to the number of GPUs available to us. 
# Dask is relevant in the world of GPU computing and RAPIDS makes it very easy to use Dask with cuDF and XGBoost.

# We'll define a function called add_5_to_x that takes some value x and adds 5 to it.

def add_5_to_x(x):
    return x + 5


def sleep_1():
    time.sleep(1)
    return 'Success!'


def main():
    # We need to setup a Local Cluster of workers to execute our work 
    # Also, we need to setup a Client to coordinate and schedule work for that cluster. 
    # As we see below, we can inititate a cluster and client using only few lines of code.
    # Here, we create a local cluster with 4 workers
    n_workers = 4
    cluster = LocalCluster(n_workers=n_workers)
    client = Client(cluster)
    # Let's inspect the client object to view our current Dask status. 
    # We should see the IP Address for our Scheduler as well as the the number of workers in our Cluster.
    # show current Dask status
    print("\nclient:\n", client, flush=True)
    # Next, we'll iterate through our n_workers and create an execution graph, 
    # Each worker is responsible for taking its ID and passing it to the function add_5_to_x. 
    # The worker with ID 2 will take its ID and pass it to the function add_5_to_x, resulting in the value 7.
    addition_operations = [delayed(add_5_to_x)(i) for i in range(n_workers)]
    print("\naddition_operations:\n", addition_operations, flush=True)
    print("\nThe above output shows a list of several Delayed objects!\n", flush=True)
    # An important thing to note is that the workers aren't actually executing these results!
    # we're just defining the execution graph for our client to execute later.
    # The delayed function wraps our function add_5_to_x and returns a Delayed object.
    # This ensures that this computation is in fact "delayed" and not executed on the spot (lazily evaluation).
    # Next, let's sum each one of these intermediate results.
    # We can accomplish this by wrapping Python's built-in sum function.
    # We use our delayed function and store this in a variable called total.
    total = delayed(sum)(addition_operations)
    print("\ntotal:\n", total, flush=True)
    print("\nThe above output shows again a Delayed objects!\n", flush=True)
    # As we mentioned before, none of these results - intermediate or final - have actually been compute. 
    # We can compute them using the compute method of our client.
    addition_futures = client.compute(addition_operations, optimize_graph=False, fifo_timeout="0ms")
    print("\naddition_futures (i.e. client.compute on addition_operations):\n", addition_futures, flush=True)
    print("\nWe can see from the above output that our addition_futures variable is a list of Future objects\n", flush=True)
    # It is not the "actual results" of adding 5 to each of [0, 1, 2, 3]. 
    # These Future objects are a promise that at one point a computation will take place.
    # So that we will be left with a result.
    # Dask is responsible for fulfilling that promise by delegating that task to a Dask worker and collecting the result.
    # Let's take a look at our total_future object:
    total_future = client.compute(total, optimize_graph=False, fifo_timeout="0ms")
    wait(total_future)  # this will give Dask time to execute the work
    print("\ntotal_future     (i.e. client.compute on total and then wait):\n", total_future, flush=True)
    print("\nType of total_future:\n", type(total_future), flush=True)
    print("\ntotal_future is a Future with status of the request, type of the result, and a key associated with the operation\n", flush=True)
    # To collect and print the result of each of these Future objects, we can call the result() method
    addition_results = [future.result() for future in addition_futures]
    print('\nAddition Results (calling result() on each element of addition_futures):', addition_results,  flush=True)
    print("\nNow we see the results that we want from our addition operations.\n",  flush=True) 
    # We can also use the simpler syntax of the client.gather method to collect our results.
    addition_results = client.gather(addition_futures)
    total_result = client.gather(total_future)
    print('\nAddition Results (client.gather on addition_futures):\n', addition_results, flush=True)
    print('\nTotal Result: (client.gather on total_future):\n', total_result, flush=True)
    # Awesome! We just wrote our first distributed workflow.
    # To confirm that Dask is truly executing in parallel
    print("\nConsider a function that sleeps for 1 second and returns the string \"Success!\"\n", flush=True)
    print("\nIn serial, this function should take our 4 workers around 4 seconds to execute.\n", flush=True)
    start = time.time()
    for _ in range(n_workers):
        sleep_1()
    dt = time.time() - start
    print(f"\nAs expected, our process takes {dt} seconds to run\n", flush=True)
    print("\nNow let's execute this same workflow in parallel using Dask\n", flush=True)
    # define delayed execution graph
    start = time.time()
    sleep_operations = [delayed(sleep_1)() for _ in range(n_workers)]
    # use client to perform computations using execution graph
    sleep_futures = client.compute(sleep_operations, optimize_graph=False, fifo_timeout="0ms")
    # collect and print results
    sleep_results = client.gather(sleep_futures)
    dt = time.time() - start
    print(sleep_results, flush=True)
    print(f"\nUsing Dask, we see that this whole process takes {dt} - each worker is executing in parallel!\n", flush=True)
    client.close()

if __name__ == "__main__":
    main()

