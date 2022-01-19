//Enrique Tapia
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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
	
	//allocate memory for arrays using single precision floating points
	float *a = (float*)malloc(i * i * sizeof(float));
	float *b = (float*)malloc(i * i * sizeof(float));
	float *c = (float*)malloc(i * i * sizeof(float));
	float *result = (float*)malloc(i * i * sizeof(float));

	//matrix multiplication
	faxpyu(i, a, b, c, result);

	//free memory
	free(a);
	free(b);
	free(c);
	free(result);

	return 0;

}