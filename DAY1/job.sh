#!/bin/bash
#SBATCH --job-name=my_first_job
##SBATCH --output=job.out 
##SBATCH --error=job.err
#SBATCH --time=00:00:30 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=10GB
#SBATCH --partition=dcgp_usr_prod
#SBATCH --account=tra25_ictp_rom_0

sleep 600
hostname 
