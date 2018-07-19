CC=gcc
CFLAGS=-mavx -std=c99 -O0
HELLOOUT=hello_avx.out

all: hello_avx

hello_avx: hello_avx.o
	$(CC) $(CFLAGS) -o $(HELLOOUT) hello_avx.o

hello_avx.o: hello_avx.c
	$(CC) $(CFLAGS) -c hello_avx.c

clean: 
	rm *.o *.out
