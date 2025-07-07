import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

if __name__=="__main__":
    assert dist.is_available()
    dist.init_process_group("nccl")
    rank=dist.get_rank()
    # Print out the detected devices
    if rank==0:
       print("Available CUDA devices:")
       for i in range(torch.cuda.device_count()):
          print(f"Device {i}: {torch.cuda.get_device_name(i)}")

    # Bind the current process to a device, use the classical rank % device_id
    device = f"cuda:{rank % torch.cuda.device_count()}"
    print(f"Rank {rank} using device: {device}")
    torch.cuda.set_device(device)

    # Init a tensor of size 10 and move it to the device
    tensor=torch.zeros(10).to(device)
    if rank == 0:
        tensor += 42
    dist.broadcast(tensor,src=0)
    print(f"Rank {rank} received: {tensor}\n")
    dist.destroy_process_group()
