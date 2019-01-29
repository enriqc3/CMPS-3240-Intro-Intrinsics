#include <immintrin.h>
// If windows: #include <x86intrin.h>
#include "myblas.h"

/* DGEMM. From Computer Organization and Design, Patterson and Hennesey 5e
 * An unoptimized version of a double precision matrix multiplication, widely
 * known as 'DGEMM' for double-precision general matrix multiply (GEMM). 
 * Assumes matrixes are square and of the same length ('n' is the length of 
 * one side).
 *
 * Uses AVX intrinsics.
 */
void dgemmo( int n, double* A, double* B, double* C ) {
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

// Unoptimized version
void dgemmu( int n, double* A, double* B, double* C ) {
    for ( int i = 0; i < n; i++ ) {
        for ( int j = 0; j < n; j++ ) {
            double cij = C[ i + j * n ]; 

            for ( int k = 0; k < n; k++) {
                cij += A[ i + k * n ] * B[ k + j * n ];
            }

            C[ i + j * n ] = cij; 
        }
    }
}

// DAXPY operation. Not optimized with AVX intrinsics.
void daxpyu( int n, double A, double* x, double* y, double* result ) {
    for( int i = 0; i < n; i++ )
        result[i] = A * x[i] + y[i];
} 
