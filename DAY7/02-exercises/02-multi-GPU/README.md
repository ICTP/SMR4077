#  Let's go in parallel

Now it’s time to parallelize our model using **data parallelism**, which can be summarized as follows:

1. Each **rank** (i.e., process) instantiates the same model.
2. At each training iteration, data is split across ranks (only indices are coordinated—no actual dataset transfer).
3. Each rank performs the **forward** and **backward** passes on its local batch.
4. Gradients are **averaged** across all ranks using a reduction operation.
5. Each rank updates its local model weights. By design, all models remain synchronized and identical across devices.

---

## Dataset and Model

We’ll use the **CIFAR-10** dataset, which contains 60,000 RGB images (32×32 pixels) across 10 classes. The model is a **convolutional neural network (CNN)** implemented in `models.py`.



## DData Distribution

We won’t send actual data between ranks. Instead, we coordinate which data subset each rank processes. This is handled via `DistributedSampler`, which ensures each rank gets a unique shard of the dataset:

```python
train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)
```

These samplers are passed to the data loaders:

```python
train_loader = DataLoader(
    train_dataset,
    shuffle=False,
    sampler=train_sampler,
    batch_size=512 // world_size,
    num_workers=1,
    drop_last=True,
    pin_memory=True
)
```

**Notes:**

* `pin_memory=True` enables faster memory transfer from host to device.
* Batch size is divided across ranks, so each process handles a portion of the total batch.

---

##  Model Initialization

Model weights are randomly initialized. To ensure all ranks start with the **same weights**, we broadcast the model parameters from **rank 0** to all other ranks:

```python
# Pseudocode:
# for each parameters in model.parameters()
#    broadcast from rank 0 to everyone

```

---

##  Gradient Synchronization

After each rank computes its gradients locally (based on its mini-batch), we **synchronize** gradients using `all_reduce` to average them across all processes:

```python
         # Pseudocode:
         # for each parameters in model.parameters()
         #    reduce parameter.grad.data using SUM operation
         #    Calculate the average
```

This ensures the model update reflects the combined learning signal from all data partitions.

---

##  Model Evaluation

Since CIFAR-10 is relatively small, we evaluate the model after each epoch. Evaluation can also be done in parallel. Each rank computes its local accuracy, and we aggregate results using a **reduce** operation over:

* The number of correctly classified samples.
* The total number of samples evaluated.

##  Running with `torchrun`

We’ll use `torchrun` to launch the distributed training:

* `torchrun` will spawn **one process per GPU**, assigning each one a unique **NCCL rank**.
* This approach works **on a single node** with multiple GPUs.
* For multi-node setups, additional orchestration (e.g., with SLURM or Ray) is needed.

```bash
torchrun --nproc_per_node=NUM_GPUS your_training_script.py
```

You can use the script `submit.sh` provided to run you code. 

## Exercise

Solve 3 `TODO`s:
1. Synchronize the model across ranks before staring the training loop
2. Synchronize the epoch after each batch
3. Calculate the global performance
