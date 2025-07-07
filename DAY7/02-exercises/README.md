# Practical part

## Dataset setup

On the **Leonardo login node**, set up the datasets by running:

```bash
$ bash setup_datasets.sh
```

This script will download two datasets: **MNIST** and **CIFAR-10**, and place them in the `data/` folder.

This step is required because compute nodes do **not** have internet access.

---

## Single GPU

This section demonstrates how to port the training code from the notebook `01-mnist-training.ipynb` to run on a **single GPU**, using the small MNIST dataset.

## NCCL Examples

A collection of simple scripts that test and demonstrate **NCCL functionality** in PyTorch. These examples illustrate core communication primitives used in distributed GPU training.

## Multi-GPU

You will implement and parallelize the training of a model using the **data parallelism** paradigm across multiple GPUs.

## Multi-Node

In this section, you'll take the multi-GPU code you developed earlier and **scale it to run across multiple nodes** in a distributed environment.

## Distributed Data Parallel

This folder contains the parallelized version of the code originally written for training a model on a single GPU.
