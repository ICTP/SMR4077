On for example, 

mcelori1@login01

lsof -ti:8998 | xargs kill -9

and we

srun --qos=boost_qos_dbg --nodes=1 --ntasks-per-node=4 --cpus-per-task=8 --gres=gpu:4 -p boost_usr_prod --mem=450GB --time 00:30:00 --pty bash

Suppose that the outcome of 

squeue --me

is

             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
          16847142 boost_usr     bash mcelori1  R      14:25      1 lrdn1450


Then enter

module purge
module load gcc/12.2.0
module load cuda/12.1
module load openmpi/4.1.6--gcc--12.2.0
module load nvhpc/23.5
module load anaconda3/2023.09-0

conda activate /leonardo/pub/userinternal/mcelori1/MagureleRAPIDS/rapids_venv


Finally

jupyter-lab --port=8998 --no-browser

On another shell on local PC

ssh -L 8998:localhost:8998 mcelori1@login01-ext.leonardo.cineca.it ssh -L 8998:localhost:8998 -N lrdn1450

Finally open a browser and

http://localhost:8998/lab?token=68e650f4b595ce40ce74e7e0374915108ddbe589e5c41014

