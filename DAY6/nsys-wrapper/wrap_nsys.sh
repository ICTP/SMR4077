#!/bin/bash

if [[ $SLURM_PROCID == 0 ]]; then
        nsys profile -t mpi,nvtx,cuda --stats=true $*
else
	$*
fi
