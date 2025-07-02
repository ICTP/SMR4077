### Note: 

to compile the code with `openmp`, use the `-fopenmp` (if you are using gcc), like

```bash
gcc -fopenmp hello_openmp.c -o hello_openmp.x
```

### Exercise 0:

make sure you understand the examples on the slides, run them to check that they work;


### Exercise 1:

Take your MPI code for the calculation of `\pi` and add `openmp` to it. 

Once it produces a right result on 2 cores and 2 threads, run the scaling varying the number of threads and cores (in your job script, change the parameters for `ntasks-per-node` and `cpus-per-task`, control the number of threads with `OMP_NUM_THREADS` environmental variable).


### Exercise 2:

Write an openmp version of matrix multiplication. The code for the serial matrix multiplication is given to you below (make sure you understand what it does), implement different parallel functions inside of it (for example, try just adding `parallel for`, try `collapse`, try different scheduling, etc). 

Check the scaling of the code with the number of threads for the matrix sizes 100 and 2000. 

```c

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <omp.h>


#define N 100

void matrix_multiply_sequential(double* A, double* B, double* C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += A[i*n + k] * B[k*n + j];
            }
            C[i*n + j] = sum;
        }
    }
}



void initialize_matrices(double* A, double* B, int n) {
    srand(42); // For reproducible results
    for (int i = 0; i < n*n; i++) {
        A[i] = (double)rand() / RAND_MAX;
        B[i] = (double)rand() / RAND_MAX;
    }
}

double benchmark_function(void (*func)(double*, double*, double*, int),
                         double* A, double* B, double* C, int n, const char* name) {
    memset(C, 0, n*n*sizeof(double)); // Clear result matrix

    double start = omp_get_wtime();
    func(A, B, C, n);
    double end = omp_get_wtime();

    double time_taken = end - start;
    printf("%s: %.3f seconds\n", name, time_taken);
    return time_taken;
}


int main() {
    int num_threads = omp_get_max_threads();
    printf("Matrix size: %dx%d\n", N, N);
    printf("Number of threads: %d\n\n", num_threads);

    double* A = (double*)malloc(N * N * sizeof(double));
    double* B = (double*)malloc(N * N * sizeof(double));
    double* C = (double*)malloc(N * N * sizeof(double));

    if (!A || !B || !C) {
        printf("Memory allocation failed!\n");
        return 1;
    }

    initialize_matrices(A, B, N);

    // Benchmark all functions 
    double seq_time = benchmark_function(matrix_multiply_sequential, A, B, C, N,
                                       "Sequential");

    // Write results to file for scaling analysis
    FILE* fp = fopen("scaling_results.dat", "a");
    if (fp) {
        // Format: threads sequential "YOUR FUNCTIONS HERE"
        fprintf(fp, "%d %.6f \n",
                num_threads, seq_time);
        fclose(fp);
    }

    free(A); free(B); free(C);
    return 0;
}


```


