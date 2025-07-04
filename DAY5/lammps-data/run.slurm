#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=hello
#SBATCH --time=00:30:00
#SBATCH -o stdout.%j
#SBATCH --partition=boost_usr_prod

source ~/.bashrc
m="janus"
micromamba activate $m

module load openmpi/4.1.6--nvhpc--24.3 fftw/3.3.10--openmpi--4.1.6--nvhpc--24.3  nvhpc/24.3   gcc/12.2.0
module list





export LD_LIBRARY_PATH=$HOME/micromamba/envs/$m/lib/python3.12/site-packages/cuequivariance_ops/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/micromamba/envs/$m/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=xxx/lammps-mace/lib64/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/micromamba/envs/$m/lib/:$LD_LIBRARY_PATH

#mace_create_lammps_model --format mliap --dtype float64 --head r2scan ../models/zif8-medium-matpes-v7.model 
export OMP_NUM_THREADS=1
#export CUDA_VISIBILE_DEVICES=1,0,3,2
lmp="xxx/bin/lmp"

srun $lmp -k on t 1 g 1 -sf kk -pk kokkos newton on neigh half   -in in.nacl -log gpu.1
