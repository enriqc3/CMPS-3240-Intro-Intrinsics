CC=gcc
CFLAGS=-Wall -std=c99 -O0 -msse -msse2 -msse3 -mfpmath=sse
AVXOUT=avx_dgmm.out
BINEXT=out

all: hello_sse fgemmu fgemmo

# Target to create our BLAS library
myblas.o:   myblas.c hello_sse
	$(CC) $(CFLAGS) -c $< -o $@

# Functions to test if AVX exists on a processor by running the AVX 
# ... instructions.
hello_sse: hello_sse.o
	$(CC) $(CFLAGS) -o $@.$(BINEXT) $^
hello_sse.o: hello_sse.c
	$(CC) $(CFLAGS) -c $< -o $@

# Target to create test function for unoptimized version of DGEMM
fgemmu: fgemmu.o myblas.o
	$(CC) $(CFLAGS) -o $@.$(BINEXT) $^
fgemmu.o: fgemmu.c
	$(CC) $(CFLAGS) -c $< -o $@

# Target to create test function for optimized version of DGEMM
fgemmo: fgemmo.o myblas.o
	$(CC) $(CFLAGS) -o $@.$(BINEXT) $^
fgemmo.o: fgemmo.c
	$(CC) $(CFLAGS) -c $< -o $@

clean: 
	rm -f *.o *.out
