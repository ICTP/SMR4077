#ifdef _OPENACC
#include <accel.h>              // OpenACC
#endif
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <mpi.h>

#ifdef HDF5
#include "hdf5.h"
#define H5FILE_NAME     "jacobi.h5"
#define DATASETNAME 	"diffusion" 
#define RANK   2
#endif



/*** function declarations ***/
int calculate_nloc(int tot_col,int size,int rank) {
  //calculate how many row belong to the current rank
  return (rank < tot_col % size) ? tot_col/size +1 : tot_col/size;
}


int calculate_offset(int size,int tot_col,int rank) {
  //calculate the offset for each block
  int n_resto = (tot_col / size) + 1;
  int n_normale = (tot_col / size);
  int diff = rank - (tot_col % size);
  return (rank < tot_col % size) ? n_resto*rank  : n_resto * (tot_col % size) + n_normale * diff ;
}

void set_recvcout(int* recvcount, int size,int N){
  //set the recv_count array
  for(int p=0;p<size;++p){  
    recvcount[p]=calculate_nloc(N,size,p)*(N+2);
  }
  recvcount[0]+=N+2;
  recvcount[size-1]+=N+2;
}

void set_displacement(int* displacement,const int* recvcount,int size) {
  //calculate the displacement array using the recv_count array
  displacement[0]=0;
  for(int p=1;p<size;++p)
    displacement[p]=displacement[p-1]+recvcount[p-1];
}


void save_hdf(int dimension,int image_id,int nloc,MPI_Comm comm,double* matrix) ;

// save matrix to  .dat file in order to render with gnuplot
void save_gnuplot( double *M, size_t dim );

// evolve Jacobi
void evolve( double * matrix, double *matrix_new, size_t dimension, size_t nloc );

// return the elapsed time
double seconds( void );

/*** end function declaration ***/

int main(int argc, char* argv[]){

  // MPI init env
  MPI_Init(&argc,&argv);
  int size,rank;
  MPI_Comm_size(MPI_COMM_WORLD,&size);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);

  double comm_time=0,compute_time=0;
#ifdef _OPENACC
  const int num_dev = acc_get_num_devices(acc_device_nvidia);  // #GPUs
  const int dev_id = rank % num_dev;         
  acc_set_device_num(dev_id,acc_device_nvidia); // assign GPU to one MPI process
  acc_init(acc_device_nvidia);                                 // OpenACC call
  printf("Rank %d/%d, device %d/%d\n",rank,size,dev_id,num_dev);
#endif

  // timing variables
  double increment;

  // indexes for loops
  size_t i, j, it;
  
  // initialize matrix
  double *matrix, *matrix_new, *tmp_matrix;

  size_t dimension = 0, iterations = 0;
  size_t byte_dimension = 0;

  // check on input parameters
  if(argc != 3) {
    fprintf(stderr,"\nwrong number of arguments. Usage: ./a.out dim it\n");
    return 1;
  }

  dimension = atoi(argv[1]);
  iterations = atoi(argv[2]);


  if(rank==0) {
    printf("matrix size = %zu\n", dimension);
    printf("number of iterations = %zu\n", iterations);
  }


  size_t nloc=dimension/size;
  if(rank < (dimension % size))
    nloc++;

  byte_dimension = sizeof(double) * ( nloc + 2 ) * ( dimension + 2 );
  matrix = ( double* )malloc( byte_dimension );
  matrix_new = ( double* )malloc( byte_dimension );

  memset( matrix, 0, byte_dimension );
  memset( matrix_new, 0, byte_dimension );

  //fill initial values  
  for( i = 1; i <= nloc; ++i )
    for( j = 1; j <= dimension; ++j )
      matrix[ ( i * ( dimension + 2 ) ) + j ] = 0.5;

  // set up borders 
  increment = 100.0 / ( dimension + 1 );
  int offset = calculate_offset(size,dimension+2,rank);  
  for( i=1; i <= nloc+1; ++i ){
    //Init vertical borborder
    matrix[ i * ( dimension + 2 ) ] = (i+offset) * increment;
    matrix_new[ i * ( dimension + 2 ) ] = (i+offset)  * increment;
  }

  if(rank==(size-1))
    //The last process init also the horizontal border
    for( i=1; i <= dimension+1; ++i ){
      matrix[ ( ( nloc + 1 ) * ( dimension + 2 ) ) + ( dimension + 1 - i ) ] = i * increment;
      matrix_new[ ( ( nloc + 1 ) * ( dimension + 2 ) ) + ( dimension + 1 - i ) ] = i * increment;
    }

  // Start algorithm
  int width=(dimension+2);
  int image=0;

#pragma acc enter data copyin(matrix[:(dimension+2)*(nloc+2)],matrix_new[:(dimension+2)*(nloc+2)])
  for( it = 0; it < iterations; ++it ){

    //send up,recv bottom
    int send_to=  (rank-1)>=0 ? rank-1 : MPI_PROC_NULL;
    int recv_from= (rank+1)<size ? rank+1 : MPI_PROC_NULL;
    double time = seconds();
    
 #pragma acc host_data use_device(matrix)
 { 

    MPI_Sendrecv(matrix+ width, width, MPI_DOUBLE,
                 send_to, 0,
                 matrix+width*(nloc+1), dimension+2, MPI_DOUBLE,
                 recv_from, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(matrix+width*nloc, width, MPI_DOUBLE,
                 recv_from, 0,
                 matrix, width, MPI_DOUBLE,
                 send_to, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
 }
 comm_time+=seconds()-time;

time=seconds();
#pragma acc  data present(matrix[:(dimension+2)*(nloc+2)],matrix_new[:(dimension+2)*(nloc+2)]) 
 {
  #pragma acc parallel loop collapse(2)
   for(int i = 1 ; i <= nloc; ++i )
     for(int j = 1; j <= dimension; ++j )
       matrix_new[ ( i * ( dimension + 2 ) ) + j ] = ( 0.25 ) * 
	 ( matrix[ ( ( i - 1 ) * ( dimension + 2 ) ) + j ] + 
	   matrix[ ( i * ( dimension + 2 ) ) + ( j + 1 ) ] + 	  
	   matrix[ ( ( i + 1 ) * ( dimension + 2 ) ) + j ] + 
	   matrix[ ( i * ( dimension + 2 ) ) + ( j - 1 ) ] ); 
 }
 compute_time+=seconds()-time;

#ifdef _OPENACC
  //swap the pointers on the device
#pragma acc serial present(matrix[:(dimension+2)*(nloc+2)],matrix_new[:(dimension+2)*(nloc+2)])
 {
   double* tmp_matrix = matrix;
   matrix = matrix_new;
   matrix_new = tmp_matrix;
 }

#endif

 //in order to preserve data coherency swap pointers on the host
 tmp_matrix = matrix;
 matrix = matrix_new;
 matrix_new = tmp_matrix;


#ifdef HDF5
 if((it % 100)==0) {
   image++;
   save_hdf(dimension,image,nloc,MPI_COMM_WORLD,matrix) ;
 }
#endif

}


  //Back to the host
#pragma acc exit data copyout(matrix[:(dimension+2)*(nloc+2)],matrix_new[:(dimension+2)*(nloc+2)]) 
  int* recvcount=NULL;
  int* displacement=NULL;
  double* matrix_final=NULL;
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank==0) {
    printf( "%d Computation time %f, communication time %f\n",size,compute_time,comm_time );
    recvcount=malloc(size*sizeof(int));
    displacement=malloc(size*sizeof(int));
    matrix_final=malloc((dimension+2)*(dimension+2)*sizeof(double));
    set_recvcout(recvcount,size,dimension);
    set_displacement(displacement,recvcount,size);
  }
#ifdef DUMP
  if(rank==0){
    MPI_Gatherv(matrix_new, (dimension+2)*(nloc+1), MPI_DOUBLE,
                matrix_final, recvcount, displacement,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
  } else if(rank==(size-1)) {
    MPI_Gatherv(matrix_new+(dimension+2),  (dimension+2)*(nloc+1), MPI_DOUBLE,
                NULL, NULL, NULL,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
  } else {
    MPI_Gatherv(matrix_new+(dimension+2),  (dimension+2)*(nloc), MPI_DOUBLE,
                NULL, NULL, NULL,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }

  if(rank==0) {
    save_gnuplot( matrix_final, dimension );
  }
  
#endif
  
  MPI_Finalize();
  free( matrix );
  free( matrix_new );

  return 0;
}

#ifdef HDF5
void save_hdf(int dimension,int image_id,int nloc,MPI_Comm comm,double* matrix) {
  herr_t	status;
  int rank;
  int size;
  MPI_Comm_rank(comm,&rank);
  MPI_Comm_size(comm,&size);
  /* 
   * Set up file access property list with parallel I/O access
   */
  hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);
  
  /*
   * Create a new file collectively and release property list identifier.
   */
  
  hid_t       file_id, dset_id;
  hid_t       filespace, memspace;      /* file and memory dataspace identifiers */
  char filename[256];
  snprintf(filename, sizeof(filename), "solution_%d.h5", image_id);
  file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
  H5Pclose(plist_id);
  /*
   * Create the dataspace for the dataset.
   */
  hsize_t     dimsf[2];
  dimsf[0] = dimension; //vertical
  dimsf[1] = dimension+2; //horiziontal
  filespace = H5Screate_simple(RANK, dimsf, NULL); 
  
    /*
     * Create the dataset with default properties and close filespace.
     */
    dset_id = H5Dcreate(file_id, DATASETNAME, H5T_NATIVE_DOUBLE, filespace,
                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Sclose(filespace);
    /* 
     * Each process defines dataset in memory and writes it to the hyperslab
     * in the file.
     */
    hsize_t	count[2];
    count[0] =nloc; //vertical size
    count[1] =dimension+2; //horizontal size
    hsize_t offset_file[2];

    offset_file[0]=0;
    for(int i=0;i<rank;++i)
      offset_file[0]+= calculate_nloc(dimension,size,i);
    offset_file[1] = 0; //horizontal offset
    //printf("\nOffset rank %d, offset %d, nloc %d\n",rank,offset_file[0], nloc);
    memspace = H5Screate_simple(RANK, count, NULL);

    /*
     * Select hyperslab in the file.
     */
    filespace = H5Dget_space(dset_id);
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset_file, NULL, count, NULL);
    /*
     * Create property list for collective dataset write.
     */
    plist_id = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
    status = H5Dwrite(dset_id, H5T_NATIVE_DOUBLE, memspace, filespace,
                      plist_id, matrix);
    /*
     * Close/release resources.
     */
    H5Dclose(dset_id);
    H5Sclose(filespace);
    H5Sclose(memspace);
    H5Pclose(plist_id);
    H5Fclose(file_id);
}
#endif

void evolve( double * matrix, double *matrix_new, size_t dimension , size_t nloc){
  
  size_t i , j;

  //This will be a row dominant program.
  for( i = 1 ; i <= nloc; ++i )
    for( j = 1; j <= dimension; ++j )
      matrix_new[ ( i * ( dimension + 2 ) ) + j ] = ( 0.25 ) * 
	( matrix[ ( ( i - 1 ) * ( dimension + 2 ) ) + j ] + 
	  matrix[ ( i * ( dimension + 2 ) ) + ( j + 1 ) ] + 	  
	  matrix[ ( ( i + 1 ) * ( dimension + 2 ) ) + j ] + 
	  matrix[ ( i * ( dimension + 2 ) ) + ( j - 1 ) ] ); 
}

void save_gnuplot( double *M, size_t dimension){
  size_t i , j;
  const double h = 0.1;
  FILE *file;
  file = fopen( "solution.dat", "w" );
  for( i = 0; i < dimension + 2; ++i )
    for( j = 0; j < dimension + 2; ++j )
      fprintf(file, "%f\t%f\t%f\n", h * j, -h * i, M[ ( i * ( dimension + 2 ) ) + j ] );
  fclose( file );

}

// A Simple timer for measuring the walltime
double seconds(){
    struct timeval tmp;
    double sec;
    gettimeofday( &tmp, (struct timezone *)0 );
    sec = tmp.tv_sec + ((double)tmp.tv_usec)/1000000.0;
    return sec;
}

