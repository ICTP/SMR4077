#!/bin/bash -l
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --partition=lrd_all_serial
#SBATCH --time=4:00:00
#SBATCH --mem=30GB
#SBATCH --job-name=install_cupy_mpi4py_1
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

module purge
module load gcc/12.2.0
module load cuda/12.1

eval "$(${HOME}/miniforge3/bin/conda shell.bash hook)"

conda config --set channel_priority flexible

ENV="$(pwd)"/rapids_venv

conda env create -p ${ENV} --file=environment.yml
conda activate ${ENV}
conda install cuda-cudart cuda-version=12
conda install seaborn -c conda-forge
conda deactivate



