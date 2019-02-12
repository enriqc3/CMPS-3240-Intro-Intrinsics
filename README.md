# CMPS-3240-Subword-Parallelism

An introduction to subword parallelism intrinsics with the SSE instruction set

# Introduction

## Objectives

* Familiarize yourself with multimedia intrinsics
* Create a program that uses SSE instructions to implement subword parallelism
* Observe run time improvement with FGEMM
* Implement SSE intrinsics with FAXPY

## Prerequisities

* Indexing and storing a 2-D array as a 1-D array
* Vector element-wise multiplication
* Matrix multiplication
* Subword parallelism
* x86 intrinsics section in the textbook

## Requirements

### Software

This lab requires the following software: `gcc`, `make`, `git`. `odin.cs.csubak.edu` has these already installed.

### Compatability

| Linux | Mac | Windows |
| :--- | :--- | :--- |
| Yes | Yes | Untested|

This lab requires the SSE instruction set, an addition to the x86 instruction set. This was introduced with the Pentium III in 1999, so you will almost certainly have this on your Linux or Mac. The only exception is that some Macbooks (Older than Mac OS X v10.6 Snow Leopard, 2009) use PowerPC processors and won't work for this lab becaue x86 is not the same ISA as PowerPC. This lab will work for Windows but the x86 intrinsic function calls and header names are different between Linux and Windows.

### Verify SSE instruction set

As stated above, if your processor is older than 1999 and an x86, you will meet the minimum hardware requirements for this lab. You can probably skip this step, which verifies the SSE instruction set on your processor. ```hello_sse.c``` will test your machine for the SSE instruction set. It initializes then subtracts two 128-bit SSE registers. The following:

```c
__m128 evens = _mm_set_ps(2.0, 4.0, 6.0, 8.0);
__m128 odds = _mm_set_ps(1.0, 3.0, 5.0, 7.0);
```

Initializes two 128-bit SSE registers by partitioning them into 4 32-bit floating point values. Note 128/32 is 4. Also note that ```evens``` vector values are the ``odds`` vector plus one, so subtracting the two should result in a vector of ones. This code:

```c
__m128 result = _mm_sub_ps(evens, odds);
```

Carries out the subtraction. Observe that interacting with x86 intrinsics for SSE instructions requires the use of special functions that often start with ```__m128``` for the data type and ```_mm``` prefix for functions. Compile `hello_sse` and run it from the terminal like so:

```
$ make hello_sse
gcc -Wall -std=c99 -O0 -msse -msse2 -msse3 -mfpmath=sse -o hello_sse.out hello_sse.o
```

If you get:

```
Illegal instruction (core dumped)
```

your processor does not have SSE. You're on a processor older than 1999, or perhaps on a Raspberry Pi that is not x86. There is nothing that can be done. Please use ```odin.cs.csubak.edu``` instead. If it works, you should get:

```
1.000000 1.000000 1.000000 1.000000
```

and you're good to go.

### Sidebar: `pd` and `ps`

The x86 intrinsics often have either a `pd` or a `ps` suffix indicating the operation is double precision or single precision respectively. This is not required to complete the lab, but if you wanted a simple exercise for yourself modify `hello_sse.c` to be double precision rather than single precision.

### Sidebar: What happened to AVX?

If you read the book, you'll notice that they are using the AVX instruction set. The AVX instruction set was released with the Intel Sandy Bridge and AMD Bulldozer chips, and there is a real danger that the end-user will not have these ISAs yet. This is why this lab uses SSE instead. If you were to implement these AVX instructions you would need to have additional code to check if the instruction sets exist at run time, and that topic is beyond the scope of this lab.

## Background

Sub-word parallelism improves execution time of many repetitive tasks. Consider a problem where we have to add two vectors together. Each element of the vector is word length. There are registers and operations that operate on quad or greater word length. We can place four words in this over-sized register, execute an over-sized addition operation and (assuming we withheld the carry operation at the appropriate points) a single instruction can carry out four addition operations at once.

This is not a new concept. The SSE instructio nset was released around 1999 with the Intel Pentium III. These instructions are designed for graphics and multimedia applications that need high precision. Most applications do not need to run such high-precision operations, and instead use SIMD operations to carry out multiple low precision operations with a single instruction, assuming you harvest the appropriate word/result from within an over-sized MM register.

For a more in depth introduction re-read sections 3.7-3.8 in Patterson and Hennesey 5e. Once you fully understand these sections, read the following document:

For a full list of all SIMD instructions see:

* https://software.intel.com/sites/landingpage/IntrinsicsGuide/

# Approach

This lab consists of three parts: (1) Look at `myblas.c` to see an example of how SSE intrinsics are implemented, (2) run `fgemmu.out` and `fgemmo.out` to see the improvement with using SIMD, and (3) apply what you've learned to implement `fdaxpy.out` on your own. 

## Part 1 - Understand some SSE instructions with FGEMM

FGEMM stands for single precision generic matrix multiplication (F for `float`. I will not go into a matrix multiplication here, you should be familiar with it. Suffice to say when multiplying two matrixes together, there are many, *many* addition operations. `fgemmu()` and `fgemmo()` in the file `myblas.c` contain C code to multiply two matrixes together. `fgemmu.c` and `fgemmo.c` contain test code to initialize the matrixes dynamically on the heap using ```malloc```, and the size of the square matrixes is given as a command line argument. 

### Unoptimized code

Consider the following code, which is a modified version of the example given in the  Patterson and Hennesey textbook:

```c
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
```

This code carries out the matrix multiplication of matrixes A and B, and stores the result in C. The cost of the matrix multiplication is  O(n^3). Compile it, and run/time it for yourself:

```
$ make fgemmu
gcc -Wall -std=c99 -O0 -msse -msse2 -msse3 -mfpmath=sse -o hello_sse.out hello_sse.o
gcc -Wall -std=c99 -O0 -msse -msse2 -msse3 -mfpmath=sse -c myblas.c -o myblas.o
gcc -Wall -std=c99 -O0 -msse -msse2 -msse3 -mfpmath=sse -o fgemmu.out fgemmu.o myblas.o
$ time ./fgemmu.out 1024
```
On the local machines in 315 I get:

```
Running matrix multiplication of size 1024 x 1024
real	0m9.020s
user	0m9.016s
sys	0m0.004s
```

The machines in 315 are older, but they are indeed workstation grade PCs. Even so this is a slow operation for these machines.

### AVX instructions

Go back into `myblas.c`. We will now consider `fdgemmo()`, which is an optimized version of `fdgemmu()`. The first thing to notice is:

```c
#include <xmmintrin.h>
```

This header file includes the compiler intrinsics for the SIMD instruction sets. The text calls for ```#include <x86intrin.h>``` because it is written for Windows. If on Linux, use the above. Consider the following code:

```c
void fgemmo( int n, float* A, float* B, float* C ) {
    for ( int i = 0; i < n; i+=4) {
        for ( int j = 0; j < n; j++ ) {
            __m128 cij = _mm_loadu_ps(C + i + j * n);
            for ( int k = 0; k < n; k++) {
                float d = B[k+j*n];
                cij = _mm_add_ps(
                    cij, // +=
                    _mm_mul_ps(
                        _mm_loadu_ps(A + i + k * n),
                        _mm_set_ps1(d)
                    )
                );

            }
            _mm_storeu_ps(&C[ i + j * n ], cij);
        }
    }
}
```

`__m128 cij` creates a variable `cij` and associates it with a 128-bit double precision floating point number. `_mm_loadu_ps` dereferences `C[ i + j * n ]` and stores the result in `cij`. `_mm_add_ps(cij,.)` takes the place of `cij +=...` in the unoptimized code. `_mm_loadu_ps(.)` loads four successive values, whereas `_mm_set_ps1(.)` repeats the same scalar value into many positions of the oversized register. Admittedly, this is hard to read, so you may want to just write out a four by four multiplication by hand to ensure that all the terms are there after fully expanding everything. Note that on windows, you would use `_mm_load_pd(.)` in place of `_mm_loadu_pd(.)`. The same goes for `store`. The reason for this is that POSIX and Windows systems organize words differently in memory. Go ahead and compile and time the code:

```
$ make fgemmo
gcc -Wall -std=c99 -O0 -msse -msse2 -msse3 -mfpmath=sse -o hello_sse.out hello_sse.o
gcc -Wall -std=c99 -O0 -msse -msse2 -msse3 -mfpmath=sse -c myblas.c -o myblas.o
gcc -Wall -std=c99 -O0 -msse -msse2 -msse3 -mfpmath=sse -o fgemmo.out fgemmo.o myblas.o
$ time ./fgemmo.out 1024
Running matrix multiplication of size 1024 x 1024
real	0m2.738s
user	0m2.734s
sys	0m0.004s
```

So there is nearly a three-times improvement when using SSE instructions.

## Part 2 - Implement DAXPY with and without AVX

In part 2 we will use a different operation, double-precision constant times a vector plus a vector (DAXPY):

D = a * X + Y

Where D, X and Y are vectors (not matrixes this time) of the same size, and a is a scalar. The C code looks like so:

```c
void daxpy( int m, double A, double* X, double* Y, double* result ) {
    for ( int i = 0; i < m; i++ ) {
        result[ i ] = A * X[ i ] + Y[ i ];
    }
}
```

Implement an unoptimized version of DAXPY first, then make sure it works. Some tips:

* When using `malloc` you need to allocate only `N` space rather than `N*N` space
* Note that DAXPY has three input arguments. The scalar fractional number, X and Y.

Now use AVX intrinsics to speed it up. Some tips:

* The profile for your function should look like:
```c
void daxpy( int m, const double* A, double* X, double* Y, double* result );
```
note that the `const double* A`. x86 intrinsics can broadcast a double into packed mm positions but you must pass it the pointer not the value itself.
* You can do this in one function call and a `_mm256_storeu_pd`. It should look like:
```
result <- add( multiply( scalar, load from X ), load from Y )
store( result )
```
check `avx_dgmm.c` for how to implement this.
* `_mm256_broadcast_sd( const double* value )` is the function to broadcast the same scalar double precision value into four positions in an mm register. Note that you must pass it the pointer to the value, not the value itself.
* The vectors need to be very large for you to see differences in timings. I recommend at least 2^27 = 134217728. 

You should get a modest improvement. Some results that I got:

```shell
Albert@Alberts-MacBook-Pro:~/CMPS-3240-Subword-Parallelism$ time ./unopt_daxpy.out 100000000
Running vector addition of size 100000000 x 1
real	0m2.661s
user	0m1.962s
sys	0m0.692s
Albert@Alberts-MacBook-Pro:~/CMPS-3240-Subword-Parallelism$ time ./avx_daxpy.out 100000000
Running vector addition of size 100000000 x 1
real	0m2.624s
user	0m1.920s
sys	0m0.697s
```

# Discussion

Include responses to the following questions in your lab report:

1. If you're using 256-bit sized AVX registers to hold 64-bit sized floating point numbers, what will happen to the DGMM code if N is not a multiple of 4.
2. What factor improvement did you achieve? Does this make sense? E.g., if using 256-bit sized AVX registers to hold 64-bit sized floating point numbers one would think that there should be a four fold improvement. What factors are preventing this?

# References

* `hello.avx` was taken from https://www.codeproject.com/Articles/874396/Crunching-Numbers-with-AVX-and-AVX
