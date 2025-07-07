SHELL = /bin/sh
CC = mpicc
GCC = gcc

jacobi_cpu:
	${CC} jacobi.c -O3 -o ${@}.x 

jacobi_hdf5:
	${CC} jacobi.c -O3 -o ${@}.x -I ${HDF5_INC} -L ${HDF5_LIB} -lhdf5 -DHDF5 -DDUMP

jacobi_gpu:
	${CC} jacobi.c -o ${@}.x -DOPENACC -Minfo=all -acc -ta=tesla,time -fast
clean:
	rm -f *.x
