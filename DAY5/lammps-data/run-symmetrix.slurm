#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
##SBATCH --cpus-per-task=8
#SBATCH --job-name=hello
#SBATCH --time=00:10:00
#SBATCH -o stdout.%j
#SBATCH --partition=boost_usr_prod

source ~/.bashrc

module load openmpi/4.1.6--nvhpc--24.3 gcc/12.2.0


export LIBRARY_PATH=$CUDA_HOME/lib64/stubs:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/lib64/stubs:$LD_LIBRARY_PATH


export LD_LIBRARY_PATH=../lammps-symmetrix-gpu/lib:../lammps-symmetrix-gpu/lib64:$LD_LIBRARY_PATH

export OMP_NUM_THREADS=1
lmp="xxx/bin/lmp"

ldd $lmp
export OMP_PROC_BIND=spread 
export OMP_PLACES=threads

srun $lmp -k on t 1 g 1 -sf kk -pk kokkos newton on neigh half   -in in.water -log water.gpu.1
