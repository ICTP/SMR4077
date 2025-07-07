
# Communication with NCCL

We will use **collective communication** across GPUs to parallelize our model using NCCL.

## Contents

This section contains three examples:

- `00-nccl-hello-world`
- `01-nccl-broadcast`
- `02-nccl-allreduce`

---

## `00-nccl-hello-world`

This example demonstrates how to spawn multiple Python processes and initialize an NCCL communicator.

The core component is **process spawning**, which uses the `torchrun` utility:

```bash
torchrun --nproc_per_node=4 --nnodes=1 --rdzv_backend=static --master_addr=localhost --master_port=12345 main.py
```

This command spawns `--nproc_per_node` processes on the local node and initializes the NCCL communicator. At the **single-node level**, this is equivalent to `mpirun`.

To scale across multiple nodes, you must combine `srun`  with `torchrun`.

---

##  Multiple devices

Is common to map one single NCCL rank to a single device, (e.g. one per GPUs), to achieve this, we will define a variable as follow:
```
device = f"cuda:{rank % torch.cuda.device_count()}"
print(f"Rank {rank} using device: {device}")
torch.cuda.set_device(device)
```
## `01-nccl-broadcast`

This example illustrates **GPU collective communication** using a broadcast operation.

Each process performs the following steps:

1. Binds itself to a specific GPU:

   ```python
   device = f"cuda:{rank % torch.cuda.device_count()}"
   ```
2. Allocates an array on the GPU (the master process initializes it).
3. Participates in a broadcast operation with all other processes.
4. Prints the result after communication.

**Hint:** Keep this example in mind — it will be useful later!

```
Before broadcast:
       ---------------------
Gpu 0  | 42 | 42 | 42 | 42 |
       ---------------------
       ---------------------
Gpu 1  | 0  | 0  | 0  | 0  |
       ---------------------

After broadcast:
       ---------------------
Gpu 0  | 42 | 42 | 42 | 42 |
       ---------------------
       ---------------------
Gpu 1  | 42 | 42 | 42 | 42 |
       ---------------------
```
---

## `02-nccl-allreduce`

This example shows how to perform an **all-reduce** operation using PyTorch with the NCCL backend.

Each process does the following:

1. Binds to a GPU.
2. Allocates and initializes a tensor on its device.
3. Participates in a collective **sum reduction** across all processes.
4. Prints the final result after synchronization.

**Hint:** This is a key operation in data-parallel training — remember how it works!

```

Before reduce:
       ---------------------
Gpu 0  | 1  | 1  | 1  | 1  |
       ---------------------
       ---------------------
Gpu 1  | 1  | 1  | 1  | 1  |
       ---------------------

After reduce:
       ---------------------
Gpu 0  | 2  | 2  | 2  | 2  |
       ---------------------
       ---------------------
Gpu 1  | 2  | 2  | 2  | 2  |
       ---------------------
```

## Exercise

Read the codes provided and run the jobs on **Leonardo cluster**
