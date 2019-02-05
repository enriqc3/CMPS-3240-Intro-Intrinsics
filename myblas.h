// DAXPY ops. 'u' stands for unoptimized and 'o' stands for optimized. 'daxpyo' is not provided.
void daxpyu( int n, double A, double* x, double* y, double* result );
void daxpyo( int n, double A, double* x, double* y, double* result );

// DGEMM ops. 'u' stands for unoptimized and 'o' stands for optimized. 'daxpyo' is not provided.
void dgemmu( int n, double* A, double* B, double* C );
void dgemmo( int n, double* A, double* B, double* C );
