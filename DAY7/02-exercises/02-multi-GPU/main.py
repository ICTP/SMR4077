import torch
import time
import os
import sys
import torch.optim as optim
from torchvision import datasets, transforms, models
import torch.distributed as dist
from torch.utils.data import DataLoader,DistributedSampler
from model import AlexNetCIFAR,evaluate
from data import get_dataset

assert dist.is_available()
dist.init_process_group("nccl")
rank=dist.get_rank()
world_size=dist.get_world_size()
device_id = rank % torch.cuda.device_count()
print(f"Hello from rank {rank}, using device: {device_id}\n")

model = AlexNetCIFAR().to(device_id)

# TODO: model synchronizatoin
# You have to ynchronize model parameters across all processes
# In data parallelism, each process must start with the same model parameters.
# However, by default, each process initializes its model independently with random weights.
# To ensure consistency, broadcast the model parameters from rank 0 to all other ranks.
# 
# You can iterate over the model parameters using: model.parameters()
# Use the broadcast collective from the torch.distributed package.
# (The process group has already been initialized for you.)
#

# Pseudocode:
# for each parameters in model.parameters()
#    broadcast from rank 0 to everyone


train_dataset,test_dataset = get_dataset()

# Distributed sampler, it is initialized with rank and world_size. It distribute the index of the training sample to all the processes.
train_sampler = DistributedSampler(train_dataset,num_replicas=world_size,rank=rank)
test_sampler = DistributedSampler(test_dataset,num_replicas=world_size,rank=rank)

train_loader = DataLoader(train_dataset, shuffle=False,
                              sampler=train_sampler,batch_size=512//world_size,num_workers=1, drop_last=True,pin_memory=True)

test_loader = DataLoader(test_dataset, shuffle=False,
                             sampler=test_sampler, drop_last=True)

# Define the cross entropy loss
loss = torch.nn.CrossEntropyLoss()
# Use the adam optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
async_reduce=False

for epoch in range(num_epochs):
    # Sync the distributed sampler achievi a suffle
    train_sampler.set_epoch(epoch)
    test_sampler.set_epoch(epoch)
    start_time = time.time()
    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device_id), targets.to(device_id)
        optimizer.zero_grad() 
        outputs = model(inputs)
        loss_value = loss(outputs, targets)
        loss_value.backward()
        # TODO: Average gradients across all devices
        # At this point, each process holds local gradients of the loss with respect to the model weights.
        # To implement data parallelism correctly, you need to average these gradients across all processes.
        # This is done using the NCCL all-reduce collective.

        # Note:
        # - Only the `SUM` operation is supported by NCCL. To get the average, you must divide the result by the world size manually.
        # - You should iterate over `model.parameters()` as before, but this time apply `all_reduce` to `param.grad.data` (i.e., the gradients), not the parameters themselves.
        # - (Advanced) If you're comfortable with asynchronous communication, consider using `async_op=True` to overlap communication and computation.
         # Pseudocode:
         # for each parameters in model.parameters()
         #    reduce parameter.grad.data using SUM operation
         #    Calculate the average


        optimizer.step()

    walltime= time.time() - start_time
    # Optional: Evaluation (note this can be computationally expensive)
    correct, total = evaluate(model, test_loader, device_id)

    # TODO: Aggregate evaluation results across all processes
    # Currently, only rank 0 computes and reports accuracy based on its local data.
    # To compute global accuracy, you need to aggregate:
    # - The total number of correctly classified samples (`correct`)
    # - The total number of evaluated samples (`total`)
    # Use a reduce collective (e.g., dist.reduce or dist.all_reduce) to sum these values across all ranks
    # This will allow rank 0 to compute and report the overall test accuracy.


    if rank==0:
        print(f'Epoch {epoch}, Accuracy {correct/total}, Walltime per epoch: {walltime:.4f}s')
