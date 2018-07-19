#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void dgemm( int n, double* A, double* B, double* C );
void initRandMat( int m, double* A );

int main( int arg, char *argv[] ) {
   if( arg != 2 ) {
     printf( "Usage: ./unopt_dgmm.out N ... where N is the length of one size of the matrix\n"  );
     return 0;
   }
   // Initialize random
   srand( time( 0 ) );

   /* A matrix has two sizes. For these labs we assume that the matrix is square,
    * of size N x N 
    */
   const int N = atoi( argv[1] );    
   printf( "Running matrix multiplication of size %d x %d", N, N );

   /* Create three N x N double precision floating point matrixes on the heap
    * using malloc
    */
   double *A = (double *) malloc( N * N * sizeof(double) );   // First 'A' matrix
   double *B = (double *) malloc( N * N * sizeof(double) );   // Second 'B' matrix
   double *C = (double *) malloc( N * N * sizeof(double) );   // Third 'C' matrix
  
   /* The following code loads random values into the matrixes.
    * We don't need to initialize 'C' because it will hold the
    * result.
    */
   initRandMat( N, A );  // Initialize values in A
   initRandMat( N, B );  // ... safe for B

   // Carry out double-precision generic matrix multiplication 
   dgemm( N, A, B, C );

   // Free up the memory
   free( A );
   free( B );
   free( C );

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
void dgemm (int n, double* A, double* B, double* C ) {
   for ( int i = 0; i < n; i++ ) {
      for ( int j = 0; j < n; j++ ) {
         double cij = C[ i + j * n ];

         // Below it carries out cij += A[i][k] * B[k][j]
         for ( int k = 0; k < n; k++ )
            cij += A[ i + k * n ] * B[ k + j * n ];

         C[ i + j * n ] = cij;
      }
   }
}
