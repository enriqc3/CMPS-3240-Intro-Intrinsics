#include <immintrin.h>
// If windows:
// #include <x86intrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void dgemm_avx( int n, double* A, double* B, double* C );
void initRandMat( int m, double* A );

int main( int arg, char *argv[] ) {
    // ATTENTION: You should not need to modify this function for the lab
    // ... though you should study it.
    if( arg != 2 ) {
        // TODO: Might want to reject N that is not a multiple of 4
        printf( "Usage: ./dgmm N ... where N is the length of one size of the matrix\n"  );
        return 0;
    }

    // Parameter for one side of the matrix
    const int N = atoi( argv[1] );
    printf( "Running matrix multiplication of size %d x %d", N, N );

    // Seed random
    srand( time( 0 ) );
    
    // First matrix
    double *A = (double *) malloc( N * N * sizeof(double) );
    initRandMat( N, A );  // Initialize with random values
    // Second matrix
    double *B = (double *) malloc( N * N * sizeof(double) );
    initRandMat( N, B );  // Initialize with random values
    // Third matrix, for results
    double *C = (double *) malloc( N * N * sizeof(double) );
    // No init needed for C
    /* We just allocated the memory by hand. Alternatively, there are helper
     * functions to do this for us. For C11 see assigned_alloc(). For POSIX
     * see posix_memalign(). For Windows see _aligned_malloc().
     *
     * In practice when allocating mm registers you will want to use:
     * _mm_malloc( . )
     * however for this lab the initRandMat() does not use intrinsics and
     * it would cause problems.
     *
     * This is not required for the lab.
     */

    // Call matrix multiplication function with intrinsics in it
    dgemm_avx( N, A, B, C );

    /* Note that in the unoptimized example we used free() instead.
     * _mm_free() is used for mm aligned values.
     */
    _mm_free( A );
    _mm_free( B );
    _mm_free( C );

    return 0;
}

// Code to randomly initialize the matrix
void initRandMat( int n, double* A ) {
    for ( int i = 0; i < n; i++ )
        for ( int j = 0; j < n; j++ )
            A[ i + j * n ] = 2 * rand() / (double) RAND_MAX - 1; // [-1,1]
}

// DGEMM. From Computer Organization and Design, Patterson and Hennesey 5e
// An unoptimized version of a double precision matrix multiplication, widely
// known as 'DGEMM' for double-precision general matrix multiply (GEMM). 
// Assumes matrixes are square and of the same length ('n' is the length of 
// one side).
void dgemm_avx (int n, double* A, double* B, double* C ) {
    for ( int i = 0; i < n; i+=4) {
        for ( int j = 0; j < n; j++ ) {
            //double cij = C[ i + j * n ]; replaced with:
            __m256d cij = _mm256_loadu_pd(C + i + j * n);
            /* __m256d: A 256-bit mm register with 4 doubles in it. Alt-
             * ernatively, __m256 is the type for singles and __m256i is
             * the type for integers.
             *
             * _mm256_loadu_pd loads four successive values from the
             * array into an mm register. _m256 indicates it's for
             * 256 bit registers (AVX has even larger registers!).
             * loadu indicates it's a load operation. load (rather
             * than loadu) is used for Windows. _pd indicates that the
             * operation is for double precision.
             */

            // Below carries out cij += A[i][k] * B[k][j]:
            for ( int k = 0; k < n; k++) {
                // cij += A[ i + k * n ] * B[ k + j * n ];
                // ... replaced with:
                cij = _mm256_add_pd(
                    cij, // +=
                    _mm256_mul_pd(
                        _mm256_loadu_pd(A + i + k * n),
                        _mm256_broadcast_sd(B + k + j * n)
                    )
                );
                // broadcast loads a scalar into four positions
                // load loads four sucessive values
            }
            //C[ i + j * n ] = cij; replaced with
            _mm256_storeu_pd(&C[ i + j * n ], cij);
        }
    }
}
