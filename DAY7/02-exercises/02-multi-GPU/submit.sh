#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --job-name=multi-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --gpus-per-task=2
#SBATCH --time=00:03:00

#####################################
#       ENV SETUP                   #
#####################################
module load profile/deeplrn
module load cineca-ai/

#####################################
#       RESOURCES                   #
#####################################
echo "Node allocated ${SLURM_NODELIST}"
echo "Requested ${SLURM_NNODES} nodes"
echo "Requested ${SLURM_NTASKS} tasks in total"
echo "Requested ${SLURM_TASKS_PER_NODE} task per node"
echo ""

echo "Requested ${SLURM_GPUS_ON_NODE} gpus per node"
#echo "Total gpu requested ${SLURM_GPUS}"
###################################


export LOGLEVEL=INFO

torchrun --nproc_per_node=2  --nnodes=1 --rdzv_backend=static --master_addr=localhost --master_port=12345 main.py
