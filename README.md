# CMPS-3240-Subword-Parallelism

CMPS 3240 Computer Architecture: Lab on subword parallelism

## Requirements

1. The x86 labs from hereon out assume you are using the ECE/CS departments odin server. As it turns out, good old Sleipnir does 
not support AVX instructions. 
2. AVX instruction set: We will be using the AVX instruction set that was released with Sandy bridge processors and Bulldozer processors. When using your own machine, please check that you have this.
3. Linux. The x86 instruction calls and header names are different between Linux and Windows.

## Prerequisities

* Knowledge of matrix multiplication and vector element-wise multiplication
* Knowlege of how a 2-D array can be stored and indexed as a 1-D array
* Instruction-level parallelism
* x86 intrinsics (Do the prelab)

### Verify AVX instruction set

Skip this section if you are on odin. hello_avx.c has been provided to test if the machine you're using has the AVX instruction set. It does this by attempting to run an AVX instruction. This code will attempt to initialize then subtract two 256-bit AVX registers from each other. The following lines:

```c
/* Initialize the two argument vectors */
__m256 evens = _mm256_set_ps(2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0);
__m256 odds = _mm256_set_ps(1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0);
```

Initialize two 256-bit AVX registers by partitioning them into 8 32-bit floating point values. Note that 256/32 is 8. Also note that ```evens``` is less than ```odds``` by one, so subtrating the two should result in a vector of ones. This code:

```c
/* Compute the difference between the two vectors */
__m256 result = _mm256_sub_ps(evens, odds);
```

Carries out the subtraction. Compile `hello_avx` and run it from the terminal like so:

```
make hello_avx
./hello.out
```

If, after running `./hello.out` you get the error:

```
Illegal instruction (core dumped)
```

your processor does not have AVX. There is nothing that can be done. Please use odin instead. If it works, you should get:

```
1.000000 1.000000 1.000000 1.000000 1.000000 1.000000 1.000000 1.000000
```

and you're good to go.

## Objectives

* Familiarize yourself with AVX instructions
* Create a program that uses AVX instructions to use subword parallelism to expedite a DAXPY operation
* Run an experiment to see if subword parallelism improves DAXPY execution time

# Introduction 

The general idea of this lab is that subword-parallelism can be used to improve execution time of many repetitive tasks. Consider a problem where we have to add two vectors together. Each element of the vector is word length. However, there are registers and operations that operate on quad or greater word length. We can place four words in this over-sized register, execute an over-sized addition operation and (assuming we witheld the carry operation at the appropriate points) a single instruction can carry out four addition operations at once.

With x86 processors, the instruction set that allows you to do this is called AVX. It's acronym stands for advanced vector extensions. It is also known as Sandy Bridge New Extensions, so named for the Intel chip that was the first to feature it. These instructions were designed for graphics and multimedia applications that require high precision. However, if you happen to work with numbers/registers that do not require such large registers, such as with our lab today, you can load multiple addition terms into a single AVX register and carry out a single AVX addition instruction. When doing this, the carry-bit is halted along the appropriate word boundary. This reduces multiple addition steps to a single instruction, and you can harvest the appropriate word/result from within the over-sized AVX register.

For a more in depth introduction re-read sections 3.7-3.8 in Patterson and Hennesey 5e. Once you fully understand these sections, read the following document:

* https://www.codeproject.com/Articles/874396/Crunching-Numbers-with-AVX-and-AVX

For a full list of AVX instructions see:

* https://software.intel.com/sites/landingpage/IntrinsicsGuide/

# Approach

This lab consists of two parts: (1) Run the avx_dgemm.c and avx_unopt.c code and see how much faster it is with AVX. (2) Apply what you've learned with part 1 to the DAXPY operation.

## Part 1 - Understand some AVX instructions with DGMM

DGMM stands for double precision generic matrix multiplication. I will not go into a matrix multiplication here, you should be familiar with it. Suffice to say when multiplying two matrixes together, there are many, *many* addition operations. ```unopt_dgmm.c``` contains C code to multiply two matrixes together. The matrixes are initialized dynamically on the heap using ```malloc```, and the size of the square matrixes is given as a command line argument. Consider the following code, which was taken from the  Patterson and Hennesey textbook:

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
