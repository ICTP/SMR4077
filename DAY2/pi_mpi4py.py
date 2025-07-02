#pi_numba.py
import time
from mpi4py import MPI
import numpy as np



def loop(num_steps, myrank, nprocs):
    step = 1.0/num_steps
    sum = 0.0
    for i in range(myrank, num_steps, nprocs ):
        x= (i+0.5)*step
        sum = sum + 4.0/(1.0+x*x)
    return sum


def Pi(num_steps, myrank,nprocs):
    step = 1.0/num_steps
    start=time.time()
    sum = loop(num_steps,myrank, nprocs)
    stop=time.time()
    print('tcomp for rank ', myrank, 'is ', stop-start )
    pi_loc =  np.array([sum,])*step
    pi=np.array(0., dtype=np.float64)
    start=time.time()
    comm.Reduce([pi_loc,MPI.DOUBLE],[pi,MPI.DOUBLE],op=MPI.SUM,root=0)
    stop=time.time()
    print('reduction time for rank ', myrank, 'is ', stop-start )
    if (myrank==0):
        print('The value of pi is ', pi)

if __name__ == '__main__':
    comm=MPI.COMM_WORLD
    myrank=comm.Get_rank()
    nprocs=comm.Get_size()
    
    
    Pi(100000000,myrank,nprocs)
