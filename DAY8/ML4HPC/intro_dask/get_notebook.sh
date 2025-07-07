#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:25:00
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH -p boost_usr_prod
#SBATCH -J job_notebook
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --export=NONE


#####################################
#       ENV SETUP                   #
#####################################

module purge
module load gcc/12.2.0
module load cuda/12.1
module load openmpi/4.1.6--gcc--12.2.0
module load nvhpc/23.5
module load anaconda3/2023.09-0

source $ANACONDA3_HOME/etc/profile.d/conda.sh
conda activate /leonardo/pub/userinternal/mcelori1/MagureleRAPIDS/rapids_venv


export NODE_ADDR=$( ip -4 addr show enp1s0f0 | awk '/inet / {print $2}' | cut -d/ -f1)
export NODE_PORT=12345

echo "Run on you laptop: ssh -L ${NODE_PORT}:${NODE_ADDR}:${NODE_PORT} -N $( whoami )@login.leonardo.cineca.it  -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" > connection_instruction.txt
echo "Open on you browser: http://127.0.0.1:${NODE_PORT}"  >> connection_instruction.txt
echo "Pls. do not closet the terminal just opened!"  >> connection_instruction.txt

jupyter lab --no-browser --ip "*" --port ${NODE_PORT}   --NotebookApp.token='' --NotebookApp.password=''
