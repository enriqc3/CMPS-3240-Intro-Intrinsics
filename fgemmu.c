#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "myblas.h"

int main( int arg, char *argv[] ) {
	if( arg != 2 ) {
		printf( "Usage: ./fgemmu.out N ... where N is the length of one side of the matrix\n"  );
		return 0;
	}

	// A matrix has two sides. For these labs we assume that the matrix is 
	// square, of size N x N 
	const int N = atoi( argv[1] );    
	printf( "Running matrix multiplication of size %d x %d", N, N );

	// Create three N x N single precision floating point matrixes on the 
	// heap using malloc
	float *A = (float *) malloc( N * N * sizeof(float) ); // First
	float *B = (float *) malloc( N * N * sizeof(float) ); // Second
	float *C = (float *) malloc( N * N * sizeof(float) ); // Thirst

	// Carry out double-precision generic matrix multiplication 
	fgemmu( N, A, B, C );

	// Free up the memory
	free( A );
	free( B );
	free( C );

	return 1;
}
