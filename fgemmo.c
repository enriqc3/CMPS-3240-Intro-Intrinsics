#include <immintrin.h>	// Required for '_mm_free()' 
#include <stdio.h>
#include <stdlib.h>
#include "myblas.h"

int main( int arg, char *argv[] ) {
	if( arg != 2 ) {
		printf( "Usage: ./fgemmo.out N ... where N is the length of one side of the matrix\n"  );
		return 1;
	}

	// A matrix has two sides. For this lab we assume that the matrix is 
	// square, of size N x N 
	const int N = atoi( argv[1] ); // Get size as command line arg. 
	printf( "Running matrix multiplication of size %d x %d", N, N );

	// Create three N x N single precision floating point matrixes on the 
	// heap using malloc
	float *A = (float *) malloc( N * N * sizeof(float) );   // First matrix
	float *B = (float *) malloc( N * N * sizeof(float) );   // Second matrix
	float *C = (float *) malloc( N * N * sizeof(float) );   // Third matrix

	// Carry out double-precision generic matrix multiplication 
	fgemmo( N, A, B, C );

	// Note that for the optimized version that uses SIMD ops we have to 
	// use _mm_free() and not free() 
	_mm_free( A );
	_mm_free( B );
	_mm_free( C );

	return 0;
}
