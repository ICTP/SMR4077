#include <stdio.h>
#include <string.h>
#include <sys/time.h>

int calculate_offset(int procs,int tot_col,int iter);

int calculate_col(int tot_col,int procs,int rank);

void print_matrix(double* A,int n_col,int N);

void mat_mul(double*  A,  double*  B, double* C, int n_col,int n_fix,int N );

void init_mat(double* A, int n_elem,int offset);

void rank_mat(double* A, int n_elem,int rank);

void set_recvcout(int* recvcount, int iter, int procs,int N);

void set_displacement(int* displacement,const int* recvcount,int procs) ;

void extract(double* destination ,double*source,int n_fix,int n_col,int N) ;

double myseconds();
