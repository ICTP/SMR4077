#!/bin/bash
# Example mps-wrapper.sh usage:
# > srun [...] mps-wrapper.sh -- <cmd>

TEMP=$(getopt -o '' -- "$@")
eval set -- "$TEMP"

# Now go through all the options
while true; do
    case "$1" in
        --)
            shift
            break
            ;;
        *)
            echo "Internal error! $1"
            exit 1
            ;;
    esac
done

set -u

# For each GPU in a node, start as many MPS service as visible GPUs
if [ $SLURM_LOCALID -eq 0 ]; then
    nGPUs=`nvidia-smi -L | wc -l`
    for ((i=0; i< ${nGPUs}; i++))
    do
        mkdir /tmp/mps_$i
        mkdir /tmp/mps_log_$i
        export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_$i
        export CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log_$i
        CUDA_VISIBLE_DEVICES=$i nvidia-cuda-mps-control -d
    done 
fi

# Wait for all MPS services to start
sleep 5

# Enforce appropriate pinning based on local rank
#export CUDA_VISIBLE_DEVICES=GPU_PIPE
export CUDA_VISIBLE_DEVICES=0
GPU_PIPE=$(( ${SLURM_LOCALID} % 4))
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_${GPU_PIPE}
exec "$@"
