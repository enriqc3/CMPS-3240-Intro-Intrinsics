# CMPS-3240-Subword-Parallelism

CMPS 3240 Computer Architecture: Lab on subword parallelism

## Requirements

1. The x86 labs from hereon out assume you are using the ECE/CS departments odin server. As it turns out, good old Sleipnir does 
not support AVX instructions. 
2. AVX instruction set: We will be using the AVX instruction set that was released with Sandy bridge processors and Bulldozer processors. When using your own machine, please check that you have this.

## Prerequisities

* Knowledge of matrix multiplication and vector element-wise multiplication
* Instruction-level parallelism
* x86 intrinsics (Do the prelab)

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
