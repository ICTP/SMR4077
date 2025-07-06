#!/bin/bash -l

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:25:00
#SBATCH --mem=0 
#SBATCH --exclusive 
#SBATCH -p boost_usr_prod 
#SBATCH -J run-dask-large
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err


module purge
module load gcc/12.2.0
module load cuda/12.1
module load openmpi/4.1.6--gcc--12.2.0
module load nvhpc/23.5
module load anaconda3/2023.09-0

conda activate /leonardo/pub/userinternal/mcelori1/MagureleRAPIDS/rapids_venv

JOB_OUTPUT_DIR=$(pwd)/out_large
mkdir -p $JOB_OUTPUT_DIR

export SCHEDULER_FILE="${JOB_OUTPUT_DIR}/scheduler.json"
export NODES=$(scontrol show hostnames)

export DASK_LOGGING__DISTRIBUTED=info
export DASK_DISTRIBUTED__COMM__UCX__CREATE_CUDA_CONTEXT=True
export DASK_DISTRIBUTED__COMM__UCX__CUDA_COPY=True
export DASK_DISTRIBUTED__COMM__UCX__TCP=True
export DASK_DISTRIBUTED__COMM__UCX__NVLINK=True
export DASK_DISTRIBUTED__COMM__UCX__INFINIBAND=True
export DASK_DISTRIBUTED__COMM__UCX__RDMACM=True
export UCX_MEMTYPE_REG_WHOLE_ALLOC_TYPES=cuda

# Start controller on this first task

dask scheduler --scheduler-file ${SCHEDULER_FILE} --interface "ib0" --protocol "ucx"   &

until [ -f $SCHEDULER_FILE ]
do
     sleep 1
done

echo "File found"

for NODE in ${NODES}; do
        echo "Starting ${SLURM_NTASKS_PER_NODE} workers on ${NODE}"
        for (( GPU_ID=1; GPU_ID<=${SLURM_NTASKS_PER_NODE};GPU_ID++ )); do
            srun --export=ALL,UCX_MEMTYPE_REG_WHOLE_ALLOC_TYPES=cuda,CUDA_VISIBLE_DEVICES=$(( ${GPU_ID}-1 )) \
            --cpu-bind=core -N 1 -n 1 -c ${SLURM_CPUS_PER_TASK} -w ${NODE} dask cuda worker --scheduler-file ${SCHEDULER_FILE} \
            --nthreads=${SLURM_CPUS_PER_TASK} --interface "ib0" --protocol "ucx" \
            --enable-nvlink --enable-infiniband --enable-rdmacm --rmm-pool-size "50GB" --rmm-maximum-pool-size "60GB" --death-timeout 3600  &
        done
done

until [ -f $SCHEDULER_FILE ]
do
     sleep 1
done

echo "File found"

python wait_for_workers.py --target-workers ${SLURM_NTASKS} $SCHEDULER_FILE

echo "${SLURM_NTASKS} workers are ready"

python -c "
import json
with open('$SCHEDULER_FILE') as f:
  address=json.load(f)['address']
with open('${JOB_OUTPUT_DIR}/address.dat', 'w') as g:
  g.write(address)
"

SCHED_ADDR=`cat ${JOB_OUTPUT_DIR}/address.dat`

echo "DEBUGGG ${SCHED_ADDR}"

python "/leonardo/pub/userinternal/mcelori1/MagureleRAPIDS/rapids_venv/lib/python3.12/site-packages/dask_cuda/benchmarks/local_cudf_merge.py" --scheduler-address "$SCHED_ADDR" --interface "ib0" --protocol "ucx" -c 500_000_000  --frac-match 0.3  --enable-rmm-managed --markdown  --all-to-all --threads-per-worker 8 -d 0,1,2,3,4,5,6,7  --runs 10 > $JOB_OUTPUT_DIR/raw_data.txt

echo "Client done: $(date +%s)"
sleep 2
