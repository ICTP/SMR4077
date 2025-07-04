#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --hint=nomultithread
#SBATCH -A tra25_ictp_rom
#SBATCH --time 00:05:00
#SBATCH --gres=gpu:1
#SBATCH --mem=490000MB
#SBATCH -p boost_usr_prod
#SBATCH --exclusive
#SBATCH -q boost_qos_dbg

module purge
module load nvhpc/24.3

#to compile:
#nvc -o exe filename.c -acc -gpu=cc80,cuda12.3 -Minfo=acc
#nvfortran -o exe filename.f90 -acc -gpu=cc80,cuda12.3 -Minfo=acc

# to run: 
# ./exe
