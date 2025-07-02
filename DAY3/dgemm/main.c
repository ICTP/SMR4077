#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include "auxiliary.h"
#if defined(TEST) || defined(DGEMM) || defined(DEBUG)
#include <mkl_cblas.h>
#endif


// error handling MACRO from https://gist.github.com/jefflarkin/5390993
#define cudaCheckError() {                                          \
    cudaError_t e=cudaGetLastError();				    \
    if(e!=cudaSuccess) {						\
      printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
      exit(0);								\
    }									\
  }

#define TimeIt(x)  ({		\
      cudaEvent_t start, stop;	\
      cudaEventCreate(&start);			\
      cudaEventCreate(&stop);  \
      cudaEventRecord(start, 0); \
      x;			 \
      cudaCheckError();		 \
      cudaEventRecord(stop, 0);	 \
      cudaEventSynchronize(stop);		\
      float t;						\
      cudaEventElapsedTime(&t, start, stop); cudaEventDestroy(start); cudaEventDestroy(stop); t/1000; })


#define MPITime(x) ({	\
  double t=myseconds(); \
      x;			 \
      t=myseconds()-t; t; })

  
int main(int argc,char* argv[]) {

  //init mpi environment!
  MPI_Init(&argc,&argv);
  int procs,rank;
  MPI_Comm_size(MPI_COMM_WORLD,&procs);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  
  //time stuff
  double comm_time=0;
  double copy_time=0;
  double compute_time=0.0;
  
  int N=20;
  if (argc > 1)
    N=atoi(argv[1]);
  
  //dimensione locale per il blocco relativo al rank, non cambiera mai, se sono uno dei primi mi becco il resto
  int n_fix=(rank < N % procs) ? N/procs +1 : N/procs;
  //dimensione del buffer, deve essere in grado di contenere il blocco piu grande che c'e' in giro
  int n_buffer= ( (N%procs) >= 1) ? (N/procs)+1 : N/procs ;
  //ad ogni iterazione verra' rivisto il numero di colonne
  int n_col;
  int offset;
  
  double* A= (double*) malloc(N*n_fix*sizeof(double));
  rank_mat(A,n_fix*N,rank); //ogni matrice ha il rank del possesore
  
  double* B= (double*) malloc(N*n_fix*sizeof(double));
  //rank_mat(B,N*n_fix,rank); //ogni matrice ha il rank del possesore
  offset=calculate_offset(procs,N,rank);
  memset(B, 0, N*n_fix*sizeof(double)); //inizializzo B a zero
  for(int i=0;i<n_fix;++i)
    B[offset+i*N+i]=1.0;

  double* C= (double*) malloc(N*n_fix*sizeof(double));
  memset(C, 0, N*n_fix*sizeof(double)); //inizializzo C a zero

  //allocate the buffer, it use the larger n_col possible
  double* buffer= (double*)  malloc(N*n_buffer*sizeof(double)); 
  //allocate the buffer to linearize the block
  double* square= (double*)  malloc(n_buffer*n_buffer*sizeof(double));

  //allocate displacement ad recvout array
  int* displacement = (int*)  malloc(procs*sizeof(int));
  int* recvcount = (int*)  malloc(procs*sizeof(int));

  for(int p=0;p<procs;++p){

    //numero di colonne all'iterazione corrente
    n_col=calculate_col(N,procs,p);
    set_recvcout(recvcount,p,procs,N);
    set_displacement(displacement,recvcount,procs);

    offset=calculate_offset(procs,N,p);
    MPI_Barrier(MPI_COMM_WORLD);
    comm_time+=MPITime( extract(square,B+offset,n_fix,n_col,N); );
    comm_time+=MPITime( 
    MPI_Allgatherv( square , n_col*n_fix, MPI_DOUBLE,
                    buffer ,recvcount,displacement,MPI_DOUBLE,MPI_COMM_WORLD);
			);    
        
#ifdef DGEMM
    compute_time += MPITime( cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans , n_fix , n_col , N , 1.0 , A , N , buffer , n_col , 0.0 ,  C+offset, N ); );

#else
    compute_time += MPITime( mat_mul(A, buffer, C+offset, n_col, n_fix,N); );
#endif
  }

  if(rank==0)
    printf("%d %f %f %f\n",procs,compute_time,comm_time,copy_time);

#if (defined TEST || defined DEBUG)
  // Test the program comparing the output between dgemm plain and the previous calculated output.
  double* C_final=NULL;
  double* B_final=NULL;
  double* A_final=NULL;
  double* C_final_dgemm=NULL;
    
  if(rank==0) {
      printf("Start TESTING:\n");
    //il rank 0 raccoglie tutto quanto, quindi inizializza 3 matrici NxN
    C_final=malloc(N*N*sizeof(double));
    B_final=malloc(N*N*sizeof(double));
    A_final=malloc(N*N*sizeof(double));

    //alloco una matrice NxN da calcolare cona la dgemm
    C_final_dgemm=malloc(N*N*sizeof(double));
    for(int p=0;p<procs;++p){
      recvcount[p]=N * calculate_col(N,procs,p);
    }
    set_displacement(displacement,recvcount,procs);
  }
  //Start collecting the matrices
  MPI_Gatherv(A,
              N*n_fix,
              MPI_DOUBLE,
              A_final,
              recvcount,displacement,
              MPI_DOUBLE,
              0,
              MPI_COMM_WORLD);
  MPI_Gatherv(B,
              N*n_fix,
              MPI_DOUBLE,
              B_final,
              recvcount,displacement,
              MPI_DOUBLE,
              0,
              MPI_COMM_WORLD);
  MPI_Gatherv(C,
              N*n_fix,
              MPI_DOUBLE,
              C_final,
              recvcount,displacement,
              MPI_DOUBLE,
              0,
              MPI_COMM_WORLD);

  if (rank==0) {
    //calculate the supposed correct result
    cblas_dgemm ( CblasRowMajor, CblasNoTrans, CblasNoTrans , N , N , N , 1.0 , A_final , N , B_final , N , 0.0 ,  C_final_dgemm, N );
    int flag=0;
    for(int i=0;i<N*N;++i) {
      if (abs(C_final[i] - C_final_dgemm[i]) > 1E-6) {
        //printf("Errore %f %f\n",C_final[i],C_final_dgemm[i]);
        flag++;
      }
    }
    printf("Errori rilevati: %d\n",flag);
  }

#ifdef DEBUG
  if(rank==0){
    printf("Matrice risultante:\n");
    print_matrix(C_final,N,N);
  }
#endif

#endif
  MPI_Finalize();
}
