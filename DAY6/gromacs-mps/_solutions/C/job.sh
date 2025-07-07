#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=2
#SBATCH --hint=nomultithread
#SBATCH -A tra25_ictp_rom
#SBATCH --time 00:10:00
#SBATCH --gres=gpu:4
#SBATCH --mem=490000MB
#SBATCH -p boost_usr_prod
#SBATCH --exclusive
#SBATCH -q boost_qos_dbg

module purge
ml load profile/chem-phys
ml load gromacs/2024.2--openmpi--4.1.6--gcc--12.2.0-cuda-12.2

export OMP_NUM_THREADS=2

#export GMX_GPU_PME_DECOMPOSITION=1
#export GMX_FORCE_UPDATE_DEFAULT_GPU=true
export GMX_ENABLE_DIRECT_GPU_COMM=1

gmx_mpi grompp -f pme.mdp

srun gmx_mpi mdrun -ntomp ${OMP_NUM_THREADS} -noconfout -nsteps 16000 -nstlist 300 -nb gpu -update gpu -pme gpu -npme 1 -dlb no -v
