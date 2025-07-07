#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:25:00
#SBATCH --mem=0 
#SBATCH --exclusive 
#SBATCH -p boost_usr_prod 
#SBATCH -J run-dask-pca
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --export=NONE

module purge
module load gcc/12.2.0
module load cuda/12.1
module load openmpi/4.1.6--gcc--12.2.0
module load nvhpc/23.5
module load anaconda3/2023.09-0

source $ANACONDA3_HOME/etc/profile.d/conda.sh
conda activate /leonardo/pub/userinternal/mcelori1/MagureleRAPIDS/rapids_venv


export DASK_LOGGING__DISTRIBUTED=info
export DASK_DISTRIBUTED__COMM__UCX__CREATE_CUDA_CONTEXT=True
export DASK_DISTRIBUTED__COMM__UCX__CUDA_COPY=True
export DASK_DISTRIBUTED__COMM__UCX__TCP=True
export DASK_DISTRIBUTED__COMM__UCX__NVLINK=True
export DASK_DISTRIBUTED__COMM__UCX__INFINIBAND=True
export DASK_DISTRIBUTED__COMM__UCX__RDMACM=True
export UCX_MEMTYPE_REG_WHOLE_ALLOC_TYPES=cuda
export UCX_MEMTYPE_CACHE=n

# Start controller on this first task

python pca.py

echo "Client done: $(date +%s)"
sleep 2

