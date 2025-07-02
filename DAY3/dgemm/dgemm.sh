#!/bin/bash
#SBATCH --job-name=dgemm
#SBATCH --output=job_%j_threads.out 
#SBATCH --error=job_%j_threads.err
#SBATCH --time=00:05:00 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=56
#SBATCH --mem=0
#SBATCH --partition=dcgp_usr_prod
#SBATCH --account=tra25_ictp_rom_0

module purge
module load intel-oneapi-compilers/2023.2.1
module load intel-oneapi-mkl/2024.0.0--intel-oneapi-mpi--2021.12.1

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

mpirun ./matrix_dgemm.x 8192
