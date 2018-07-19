# CMPS-3240-Subword-Parallelism

CMPS 3240 Computer Architecture: Lab on subword parallelism

## Requirements

1. The x86 labs assume you are using the ECE/CS departments ```odin.cs.csubak.edu``` server. ```sleipnir.cs.csubak.edu``` will not work because its processor is too old. If you use your own machine, see below for verifying that you have the required instruction set.  We will be using the AVX. It was released with Sandy bridge processors and Bulldozer processors. 
3. Linux. x86 intrinsic function calls and header names are different between Linux and Windows.

## Prerequisities

* Indexing and storing a 2-D array as a 1-D array
* Vector element-wise multiplication
* Matrix multiplication
* Instruction-level parallelism
* x86 intrinsics (Do the prelab)

### Verify AVX instruction set

Skip this if you are on odin. ```hello_avx.c``` will test your machine for the AVX instruction set. It initializes then subtracts two 256-bit AVX registers. The following:

```c
/* Initialize the two argument vectors */
__m256 evens = _mm256_set_ps(2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0);
__m256 odds = _mm256_set_ps(1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0);
```

Initializes two 256-bit AVX registers by partitioning them into 8 32-bit floating point values. Note 256/32 is 8. Also note that ```evens``` is greater than ```odds``` by one, so subtrating the two should result in a vector of ones. This code:

```c
/* Compute the difference between the two vectors */
__m256 result = _mm256_sub_ps(evens, odds);
```

Carries out the subtraction. Observe that interacting with x86 intrinsics for AVX instructions requires the use of special functions that often start with ```__m256``` for the data type and ```_mm256``` prefix for functions. Compile `hello_avx` and run it from the terminal like so:

```
make hello_avx
./hello.out
```

If you get:

```
Illegal instruction (core dumped)
```

your processor does not have AVX. There is nothing that can be done. Please use ```odin.cs.csubak.edu``` instead. If it works, you should get:

```
1.000000 1.000000 1.000000 1.000000 1.000000 1.000000 1.000000 1.000000
```

and you're good to go.

## Objectives

* Familiarize yourself with AVX instructions
* Create a program that uses AVX instructions to implement subword parallelism
* Observe run time improvement with DGMM
* Implement improvements with DAXPY

# Introduction 

Sub-word parallelism improves execution time of many repetitive tasks. Consider a problem where we have to add two vectors together. Each element of the vector is word length. There are registers and operations that operate on quad or greater word length. We can place four words in this over-sized register, execute an over-sized addition operation and (assuming we withheld the carry operation at the appropriate points) a single instruction can carry out four addition operations at once.

With x86 processors, the instruction set that allows you to do this is  AVX. It stands for advanced vector extensions. It is also known as Sandy Bridge New Extensions, so named for the Intel chip that was the first to feature it. These instructions were designed for graphics and multimedia applications that need high precision. If you do not need a large register you can load many terms into a single  register. Then, carry out many operations with a single instruction. When doing this, the carry-bit is along the appropriate word boundary. This reduces multiple addition steps to a single instruction, and you can harvest the appropriate word/result from within the over-sized AVX register.

For a more in depth introduction re-read sections 3.7-3.8 in Patterson and Hennesey 5e. Once you fully understand these sections, read the following document:

* https://www.codeproject.com/Articles/874396/Crunching-Numbers-with-AVX-and-AVX

For a full list of AVX instructions see:

* https://software.intel.com/sites/landingpage/IntrinsicsGuide/

# Approach

This lab consists of two parts: (1) Run the avx_dgemm.c and avx_unopt.c code and see how much faster it is with AVX. (2) Apply what you've learned with part 1 to the DAXPY operation.

## Part 1 - Understand some AVX instructions with DGMM

DGMM stands for double precision generic matrix multiplication. I will not go into a matrix multiplication here, you should be familiar with it. Suffice to say when multiplying two matrixes together, there are many, *many* addition operations. ```unopt_dgmm.c``` contains C code to multiply two matrixes together. The matrixes are initialized dynamically on the heap using ```malloc```, and the size of the square matrixes is given as a command line argument. 

### Unoptimized code

Consider the following code, which was taken from the  Patterson and Hennesey textbook:

```c
void dgemm (int n, double* A, double* B, double* C ) {
   for ( int i = 0; i < n; i++ ) {
      for ( int j = 0; j < n; j++ ) {
         double cij = C[ i + j * n ];
         // cij += A[i][k] * B[k][j]
         for ( int k = 0; k < n; k++ )
            cij += A[ i + k * n ] * B[ k + j * n ];

         C[ i + j * n ] = cij;
      }
   }
}
```

This code carries out the matrix multiplication of matrixes A and B, and stores the result in C. The cost of the matrix multiplication is actually O(n^3). If you compile it, and run/time it for yourself:

```
make unopt_dgmm
time ./unopt_dgmm.out 1024
```
you should get something along the lines of:

```
Running matrix multiplication of size 1024 x 1024
real	0m5.912s
user	0m5.900s
sys	0m0.008s
```

Even on the beefy odin server it takes almost six seconds to multiply two 1024 x 1024 sized matrixes.

### AVX instructions

Now consider ```avx_dgmm.c```. The first thing to notice is:

```c
#include <immintrin.h>
```

This header file includes the compiler intrinsics for the AVX instruction set. The text calls for ```#include <x86intrin.h>``` because it is written for Windows. If on Linux, use the above.
