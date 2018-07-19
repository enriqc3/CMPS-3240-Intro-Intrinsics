CC=gcc
CFLAGS=-mavx -std=c99 -O0
HELLOOUT=hello_avx.out
UNOPTOUT=unopt_dgmm.out
AVXOUT=avx_dgmm.out

all: hello_avx unopt_dgmm avx_dgmm

hello_avx: hello_avx.o
	$(CC) $(CFLAGS) -o $(HELLOOUT) hello_avx.o

hello_avx.o: hello_avx.c
	$(CC) $(CFLAGS) -c hello_avx.c

unopt_dgmm: unopt_dgmm.o
	$(CC) $(CFLAGS) -o $(UNOPTOUT) unopt_dgmm.o

unopt_dgmm.o: unopt_dgmm.c
	$(CC) $(CFLAGS) -c unopt_dgmm.c

avx_dgmm: avx_dgmm.o
	$(CC) $(CFLAGS) -o $(AVXOUT) avx_dgmm.o

avx_dgmm.o: avx_dgmm.c
	$(CC) $(CFLAGS) -c avx_dgmm.c

clean: 
	rm *.o *.out
