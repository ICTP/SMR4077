#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --job-name=nccl-test
#SBATCH --time=00:04:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=0
#SBATCH --gpus-per-task=4

#####################################
#       ENV SETUP                   #
#####################################
module load profile/deeplrn
module load cineca-ai/
export LOGLEVEL=INFO
#####################################
#       RESOURCES                   #
#####################################
echo "Node allocated ${SLURM_NODELIST}"
echo "Requested ${SLURM_GPUS_ON_NODE} gpus per node"
####################################
#      MASTER ELECTION             #
####################################
# Work just with newer slurm version!
#export MASTER_ADDR=$(scontrol getaddrs $SLURM_NODELIST | head -n1 | awk -F ':' '{print$2}' | sed 's/^[ \t]*//;s/[ \t]*$//')
####################################
export MASTER_ADDR=$( ip -4 addr show enp1s0f0 | awk '/inet / {print $2}' | cut -d/ -f1)
export MASTER_PORT=12345
echo "Master's ip address used ${MASTER_ADDR}"
export RANDOM=42

export LOGLEVEL=INFO

srun  torchrun \
--nnodes 2 \
--nproc_per_node 4 \
--rdzv_id ${RANDOM} \
--rdzv_backend c10d \
--rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} main.py




