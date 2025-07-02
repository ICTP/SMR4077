#!/bin/bash
#SBATCH --job-name=my_first_job
##SBATCH --output=job.out 
##SBATCH --error=job.err
#SBATCH --time=00:00:30 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBTACH --cpus-per-task=112
#SBATCH --mem=0
#SBATCH --partition=dcgp_usr_prod
#SBATCH --account=tra25_ictp_rom_0

module load opnempi

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

./prog.x 
