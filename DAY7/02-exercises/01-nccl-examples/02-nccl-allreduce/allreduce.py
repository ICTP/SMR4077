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
    if rank==0:
       print("Available CUDA devices:")
       for i in range(torch.cuda.device_count()):
          print(f"Device {i}: {torch.cuda.get_device_name(i)}")

    device = f"cuda:{rank % torch.cuda.device_count()}"
    print(f"Rank {rank} using device: {device}")
    torch.cuda.set_device(device)

    tensor=torch.ones(10).to(device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print(f"Rank {rank} received: {tensor}\n")
    dist.destroy_process_group()
