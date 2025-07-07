# Distributed training

In order to use more than one node, you can keep the same body function used in the previous exercise `02-multi-gpu`, 
What you need to adjust is the jobfile.

# `torchrun`

Torchrun is a powerfull tool to spawner$ processes and to coordinate them.
It has several features, like elastic training, so you can change at runtime the number of process parecipating to the training. In order to achieve that a coordinator is elected and each process try to have a randez vouz, the coordinator is identified by its IP address and a port, that we need to specify.

So we need to adjust the script as follow in order to get the IP address of the master.

```
$ export MASTER_ADDR=$( ip -4 addr show enp1s0f0 | awk '/inet / {print $2}' | cut -d/ -f1)
```

This is cluster dependent ! You need to know the name of the correct interface to do that, checkout the cluster documentation or ask to sysadmin.

If you have a modern version of slurm a more portable way to do that is :
```
export MASTER_ADDR=$(scontrol getaddrs $SLURM_NODELIST | head -n1 | awk -F ':' '{print$2}' | sed 's/^[ \t]*//;s/[ \t]*$//')
```

Then running just torchrun doesn't spawn processes in all nodes, byt it run just one istance of torchrun.
To do that we will combine torchrun with srun, in order to spawn one instance of torchrun per node, possibily with a cpu mask large enought to dedicate at least one core per process. 

We will run as follow:
```
srun  torchrun \
--nnodes 4 \
--nproc_per_node 4 \
--rdzv_id ${RANDOM} \
--rdzv_backend c10d \
--rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} main.py
```

The result will be:

- `srun` will spawn one instance of `torchrun` per node
- `torchrun` will spawn `nproc_per_node` processses locally
- The spawned processes will try to communicate at the addres defined in `MASTER_ADDR` to have a randez vous, create a communicator and start training.

## Scalability

The dataset used to perform this experiment is not so big as well the model, so we can deacrease the runtime by a factor of 8 using 2 nodes (strong scalability). Increasing more nodes doesn't bring any beneficial speedup. 

In order to observe better scalability, we need to run training with larger model and larger datasets where it make sense. For instance imagenet.



