# Define required macros here
SHELL = /bin/sh

CFLAGS = -Wall -O3 -Wextra
CC = mpiicc
INCLUDES = -I/opt/ohpc/pub/thor/software/imkl/2023.2.0/mkl/2023.2.0/include
LIBS = -L/opt/ohpc/pub/thor/software/imkl/2023.2.0/mkl/2023.2.0/lib/intel64 -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -qopenmp -lm

all: matrix test dgemm cuda
matrix: auxiliary.o
	${CC} main.c auxiliary.o  ${CFLAGS} ${INCLUDES} -o ${@}.x  ${LIBS}

test: auxiliary.o
	${CC} main.c auxiliary.o  ${CFLAGS} ${INCLUDES} -o matrix_${@}.x  ${LIBS}  -DTEST

dgemm: auxiliary.o
	${CC} main.c auxiliary.o  ${CFLAGS} ${INCLUDES} -o matrix_${@}.x  ${LIBS}  -DDGEMM -DTEST

auxiliary.o:
	${CC} auxiliary.c -c  ${CFLAGS} -o auxiliary.o  -qopenmp
clean:
	-rm -f *.x
