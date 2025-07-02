from mpi4py import MPI
import numpy as np

if(__name__=='__main__'):
    comm=MPI.COMM_WORLD
    myrank=comm.Get_rank()
    nprocs=comm.Get_size()
    
    if(myrank==0):
        array=np.arange(10, dtype='i')
        print('initial array on 0',array)
        comm.Send([array,10,MPI.INT],dest=1,tag=0)
        comm.Recv([array, 10, MPI.INT], source=1, tag=1)
        print('final array on 0',array)
    elif(myrank==1):
        array=np.zeros(10,dtype='i')
        comm.Recv([array, 10, MPI.INT], source=0, tag=0)
        for i in range(10):
            array[i]=2*array[i]
        print('array on 1',array)
        comm.Send([array,10,MPI.INT],dest=0,tag=1)
    MPI.Finalize()  