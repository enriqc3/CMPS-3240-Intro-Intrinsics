CC=gcc
CFLAGS=-Wall -mavx -std=c99 -O0
AVXOUT=avx_dgmm.out
BINEXT=out

all: hello_avx dgemmu dgemmo

# Target to create our BLAS library
myblas.o:   myblas.c hello_avx
	$(CC) $(CFLAGS) -c $< -o $@

# Functions to test if AVX exists on a processor by running the AVX 
# ... instructions.
hello_avx: hello_avx.o
	$(CC) $(CFLAGS) -o $@.$(BINEXT) $^
hello_avx.o: hello_avx.c
	$(CC) $(CFLAGS) -c $< -o $@

# Target to create test function for unoptimized version of DGEMM
dgemmu: dgemmu.o myblas.o
	$(CC) $(CFLAGS) -o $@.$(BINEXT) $^
dgemmu.o: dgemmu.c
	$(CC) $(CFLAGS) -c $< -o $@

# Target to create test function for optimized version of DGEMM
dgemmo: dgemmo.o myblas.o
	$(CC) $(CFLAGS) -o $@.$(BINEXT) $^
dgemmo.o: dgemmo.c
	$(CC) $(CFLAGS) -c $< -o $@

clean: 
	rm -f *.o *.out
