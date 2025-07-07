#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH -A tra25_ictp_rom
#SBATCH --time 00:30:00
#SBATCH --gres=gpu:1
#SBATCH --mem=490000MB
#SBATCH -p boost_usr_prod


module purge
module load hdf5/1.14.3--openmpi--4.1.6--nvhpc--24.3

mpirun ./jacobi_gpu_hdf5.x 5000 10000

