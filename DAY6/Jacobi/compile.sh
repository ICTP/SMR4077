#!/bin/bash
module purge
module load autoload hdf5/1.12.0--spectrum_mpi--10.4.0--binary 
make jacobi_hdf5
make jacobi_cpu

module purge
module load hpc-sdk/
make jacobi_gpu
