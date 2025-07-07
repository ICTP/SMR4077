#!/bin/bash
module purge
module load profile/deeplrn
module load cineca-ai/
echo "import torchvision
torchvision.datasets.MNIST(root='./data', train=True, download=True)
torchvision.datasets.MNIST(root='./data', train=False, download=True)
torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
torchvision.datasets.CIFAR10(root='./data', train=False, download=True)" > main.py
python main.py
rm -f main.py
module purge
