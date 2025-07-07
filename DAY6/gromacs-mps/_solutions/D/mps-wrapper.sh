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

export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
# Launch MPS from a single rank per node
if [ $SLURM_LOCALID -eq 0 ]; then
    CUDA_VISIBLE_DEVICES=0,1,2,3 nvidia-cuda-mps-control -d
fi
# Wait for MPS to start sleep 5
sleep 5

# No longer need to see other GPUs, MPS is active
export CUDA_VISIBLE_DEVICES=0
exec "$@"
