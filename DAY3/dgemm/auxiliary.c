#include <stdio.h>
#include <string.h>
#include "auxiliary.h"

int calculate_col(int tot_col,int procs,int rank) {
  //calculate how many row belong to the current rank
  return (rank < tot_col % procs) ? tot_col/procs +1 : tot_col/procs;
}

double myseconds(){
  struct timeval tmp;
  double sec;
  gettimeofday( &tmp, (struct timezone *)0 );
  sec = tmp.tv_sec + ((double)tmp.tv_usec)/1000000.0;
  return sec;
}


int calculate_offset(int procs,int tot_col,int iter) {
  //calculate the offset for each block
  int n_resto = (tot_col / procs) + 1;
  int n_normale = (tot_col / procs);
  int diff = iter - (tot_col % procs);
    return (iter < tot_col % procs) ? n_resto*iter  : n_resto * (tot_col % procs) + n_normale * diff ;
}


void print_matrix(double* A,int n_col,int N) {
  //debug purpose: print a matrix
  for(int i=0;i<n_col;++i){
    for(int j=0;j<N;++j)
      printf("%3.3g ",A[j+i*N]);
    printf("\n");
  }
}

void mat_mul(double* restrict   A,  double* restrict  B, double* restrict   C, int n_col,int n_fix,int N ) {
  for(int i=0; i<n_fix;++i) {
    register int row=i*N;
    for(int j=0;j<n_col;++j){
      register int idx=row+j;
      for(int k=0;k<N;++k) {
        C[idx]+=A[row+k]*B[k*n_col+j];
      }
    }
  }
}


void init_mat(double* A, int n_elem,int offset){
  //init the matrix sequentially
  for(int i=0;i<n_elem;++i)
    A[i]=i+offset;
}

void rank_mat(double* A, int n_elem,int rank){
  //init the matrix with the value of the rank
  for(int i=0;i<n_elem;++i)
    A[i]=rank;
}


void set_recvcout(int* recvcount, int iter, int procs,int N){
  //set the recv_count array 
  int current=calculate_col(N,procs,iter);
  for(int p=0;p<procs;++p){
    recvcount[p]=calculate_col(N,procs,p)*current;
  }

}

void set_displacement(int* displacement,const int* recvcount,int procs) {
  //calculate the displacement array using the recv_count array
  displacement[0]=0;
  for(int p=1;p<procs;++p)
    displacement[p]=displacement[p-1]+recvcount[p-1];
}

void extract(double* destination ,double*source,int n_fix,int n_col,int N) {
  //linearize the block
  for(int line=0;line<n_fix;++line) {
    memcpy( destination+n_col*line,source+N*line, n_col*sizeof(double));
  }
}
