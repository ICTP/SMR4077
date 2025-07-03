#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <openacc.h>
#include <sys/types.h>
#include <sys/time.h>

#define N 1000

double seconds()
{
    struct timeval tmp;
    double sec;
    gettimeofday( &tmp, (struct timezone *)0 );
    sec = tmp.tv_sec + ((double)tmp.tv_usec)/1000000.0;
    return sec;
}


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

    double start = seconds();
    func(A, B, C, n);
    double end   = seconds();

    double time_taken = end - start;
    printf("%s: %.3f seconds\n", name, time_taken);
    return time_taken;
}


int main() {

    printf("Matrix size: %dx%d\n", N, N);


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
        fprintf(fp, "%.6f \n", seq_time);
        fclose(fp);
    }

    free(A); free(B); free(C);
    return 0;
}


