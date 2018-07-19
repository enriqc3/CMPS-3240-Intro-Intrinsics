#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void dgemm( int n, double* A, double* B, double* C );
void initRandMat( int m, double* A );


int main( int arg, char *argv[] ) {
   if( arg != 2 ) {
     printf( "Usage: ./dgmm N ... where N is the length of one size of the matrix\n"  );
     return 0;
   }

   const int N = atoi( argv[1] );    
   // Parameter for one side of the matrix
   printf( "Running matrix multiplication of size %d x %d", N, N );

   srand( time( 0 ) );                                        // Seed random
   double *A = (double *) malloc( N * N * sizeof(double) );   // First matrix
   double *B = (double *) malloc( N * N * sizeof(double) );   // First matrix
   double *C = (double *) malloc( N * N * sizeof(double) );   // First matrix
  
   initRandMat( N, A );  // Initialize values in A
   initRandMat( N, B );  // ... safe for B
 
   dgemm( N, A, B, C );

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

         // Below is carries out cij += A[i][k] * B[k][j]
         for ( int k = 0; k < n; k++ )
            cij += A[ i + k * n ] * B[ k + j * n ];

         C[ i + j * n ] = cij;
      }
   }
}