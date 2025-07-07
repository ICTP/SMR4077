import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import torch.distributed as dist
import os
import sys
from torch.utils.data import DataLoader,DistributedSampler
import time
from torch.nn.parallel import DistributedDataParallel as DDP
from model import AlexNetCIFAR,evaluate
from data import get_dataset

assert dist.is_available()
dist.init_process_group("nccl")
rank=dist.get_rank()
world_size=dist.get_world_size()
device_id = rank % torch.cuda.device_count()
print(f"Hello from rank {rank}, using device: {device_id}\n")

model = AlexNetCIFAR().to(device_id)

for param in model.parameters():
	dist.broadcast(param.data, src=0)

train_dataset,test_dataset = get_dataset()

train_sampler = DistributedSampler(train_dataset,num_replicas=world_size,rank=rank)
test_sampler = DistributedSampler(test_dataset,num_replicas=world_size,rank=rank)

train_loader = DataLoader(train_dataset, shuffle=False,
                              sampler=train_sampler,batch_size=512//world_size,num_workers=1, drop_last=True,pin_memory=True)

test_loader = DataLoader(test_dataset, shuffle=False,
                             sampler=test_sampler, drop_last=True)

loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
async_reduce=True

for epoch in range(num_epochs):
    # Sync the distributed sampler
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
        handles=list()
        for param in model.parameters():
            if param.grad is not None:
                # All-reduce in-place
                handle = dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM, async_op=async_reduce)
                param.grad.data /= world_size
                handles.append(handle)
        if async_reduce:
           for i in handles:
               i.wait()
        optimizer.step()
    walltime= time.time() - start_time
    correct,total=evaluate(model,test_loader,device_id)
    dist.all_reduce(correct, op=dist.ReduceOp.SUM)
    dist.all_reduce(total, op=dist.ReduceOp.SUM)
    if rank==0:
        print(f'Epoch {epoch}, Accuracy {correct/total}, Walltime per epoch: {walltime:.4f}s')
