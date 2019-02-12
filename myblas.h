// FAXPY ops. 'u' stands for unoptimized and 'o' stands for optimized. 
// 'faxpyo' is not provided.
void faxpyu( int n, float A, float* x, float* y, float* result );
void faxpyo( int n, float A, float* x, float* y, float* result );

// FGEMM ops. 'u' stands for unoptimized and 'o' stands for optimized. 
void fgemmu( int n, float* A, float* B, float* C );
void fgemmo( int n, float* A, float* B, float* C );
