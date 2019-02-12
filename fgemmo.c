#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "myblas.h"

int main( int arg, char *argv[] ) {
   if( arg != 2 ) {
     printf( "Usage: ./fgemmo.out N ... where N is the length of one side of the matrix\n"  );
     return 0;
   }
   // Initialize random
   srand( time( 0 ) );

   /* A matrix has two sizes. For these labs we assume that the matrix is 
    * square, of size N x N 
    */
   const int N = atoi( argv[1] );    
   printf( "Running matrix multiplication of size %d x %d", N, N );

   /* Create three N x N single precision floating point matrixes on the heap
    * using malloc
    */
   float *A = (float *) malloc( N * N * sizeof(float) );   // First matrix
   float *B = (float *) malloc( N * N * sizeof(float) );   // Second matrix
   float *C = (float *) malloc( N * N * sizeof(float) );   // Third matrix

   // Carry out double-precision generic matrix multiplication 
   fgemmo( N, A, B, C );

   // Note that for the optimized version that uses SIMD ops
   // we have to use _mm_free rather than free
   _mm_free( A );
   _mm_free( B );
   _mm_free( C );

   return 0;
}
