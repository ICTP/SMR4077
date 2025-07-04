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


module load openmpi/4.1.6--nvhpc--23.11 cuda/12.1 gcc/12.2.0


export LD_LIBRARY_PATH=../lammps-symmetrix-gpu-121/lib:../lammps-symmetrix-gpu-121/lib64:$LD_LIBRARY_PATH

lmp="xxx/bin/lmp"


t=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$t
export OMP_PROC_BIND=spread 
export OMP_PLACES=threads
export SRUN_CPUS_PER_TASK=$t

N=1
set -x 
srun $lmp -k on t $t g $N -sf kk -pk kokkos newton on neigh half   -in in.water -log log.G-${N}_T-${t}

