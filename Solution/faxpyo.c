//Enrique Tapia
#include <immintrin.h>	// Required for '_mm_free()' 
#include <stdio.h>
#include <stdlib.h>
#include "myblas.h"

int main(int arg, char *argv[]){
	int i;
	//check command line arguments
	if(arg > 1){
		i = atoi(argv[1]);
	}
	else {
		printf("Command line arguement required\n");
		exit(1);
	}

	printf("Running matrix multiplication of size %i x %i \n", i, i);

	//create a heap array with single precision floats
	float *a = (float*)malloc(i * i * sizeof(float));
	float *b = (float*)malloc(i * i * sizeof(float));
	float *c = (float*)malloc(i * i * sizeof(float));
	float *result = (float*)malloc(i * i * sizeof(float));

	//matrix multiplication
	faxpyo(i, a, b, c, result);

	//free the heap arrays
	_mm_free(a);
	_mm_free(b);
	_mm_free(c);
	_mm_free(result);

	return 0;
}