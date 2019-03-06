#include <immintrin.h>
// If windows: #include <x86intrin.h>
#include "myblas.h"

/* DGEMM. From Computer Organization and Design, Patterson and Hennesey 5e
 * An unoptimized version of a double precision matrix multiplication, widely
 * known as 'DGEMM' for double-precision general matrix multiply (GEMM). 
 * Assumes matrixes are square and of the same length ('n' is the length of 
 * one side).
 *
 * This example differs from the book because it uses SSE intrinsics rather 
 * than AVX intrinsics. AVX intrinsics are newer and there is a risk that some
 * users might not have it. For example, 'sleipnir.cs.csubak.edu' does not have
 * AVX.
 */
void fgemmo( int n, float* A, float* B, float* C ) {
    for ( int i = 0; i < n; i+=4) {
        for ( int j = 0; j < n; j++ ) {
            //double cij = C[ i + j * n ]; replaced with:
            __m128 cij = _mm_loadu_ps(C + i + j * n);
            /* __m128: A data type that tells the compiler to put the data in
             * a 128-bit mm register, packed with four 32-bit single precision
             * floating point values. __m128d is the type for packing two 64-bit
             * double precision floating point values. __m128i is the type of 
             * packing four 32-bit integers. This notation is also used for 
             * prefixing the operations that use these types. 
             * 
             * _mm_loadu_ps loads four successive values from the array into
             * an mm register. loadu indicates it's a load operation. load
             * (rather than loadu) is used for Windows. _ps indicates that the
             * operation is for single precision. _pd indicates an operation
             * that is double precision.
             */

            // Below carries out cij += A[i][k] * B[k][j]:
            for ( int k = 0; k < n; k++) {
		float d = B[k+j*n];
                // cij += A[ i + k * n ] * B[ k + j * n ];
                // ... replaced with:
                cij = _mm_add_ps(
                    cij, // +=
                    _mm_mul_ps(
                        _mm_loadu_ps(A + i + k * n),
                        _mm_set_ps1(d)
                    )
                );
                /* _mm128_set_ps1 broadcasts a single value to four positions
                 * in a mm register. 
                 */
            }
            //C[ i + j * n ] = cij; replaced with
            _mm_storeu_ps(&C[ i + j * n ], cij);
        }
    }
}

// Unoptimized version
void fgemmu( int n, float* A, float* B, float* C ) {
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
void faxpyu( int n, float A, float* x, float* y, float* result ) {
    for( int i = 0; i < n; i++ )
        result[i] = A * x[i] + y[i];
} 
