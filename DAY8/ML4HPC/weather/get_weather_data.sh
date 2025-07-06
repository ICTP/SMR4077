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
cd    data
wget https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt
wget ftp://ftp.ncdc.noaa.gov/pub/data/ghcn/daily/by_year/20*.csv.gz


