#!/bin/bash -l
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --partition=lrd_all_serial
#SBATCH --time=4:00:00
#SBATCH --mem=30GB
#SBATCH --job-name=get_weather_data
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

mkdir data
cd data/
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz
#wget https://archive.ics.uci.edu/static/public/280/higgs.zip
gunzip HIGGS.csv.gz


