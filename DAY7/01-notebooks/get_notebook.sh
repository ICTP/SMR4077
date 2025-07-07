#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --job-name=nccl-test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=0
#SBATCH --gpus-per-task=1
#SBATCH --time=00:30:00

#####################################
#       ENV SETUP                   #
#####################################
module load profile/deeplrn
module load cineca-ai/
module load imagenet/

export NODE_ADDR=$( ip -4 addr show enp1s0f0 | awk '/inet / {print $2}' | cut -d/ -f1)
export NODE_PORT=12345

echo "Run on you laptop: ssh -L ${NODE_PORT}:${NODE_ADDR}:${NODE_PORT} -N $( whoami )@login.leonardo.cineca.it  -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" > connection_instruction.txt
echo "Open on you browser: http://127.0.0.1:${NODE_PORT}"  >> connection_instruction.txt
echo "Pls. do not closet the terminal just opened!"  >> connection_instruction.txt

jupyter lab --no-browser --ip "*" --port ${NODE_PORT}   --NotebookApp.token='' --NotebookApp.password=''
