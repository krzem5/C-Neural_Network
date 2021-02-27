#include <nn.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>



NN neural_network(uint8_t i,uint8_t h,uint8_t o,float lr){
	srand((unsigned int)time(NULL));
	NN nn=malloc(sizeof(struct __NN)+(((uint64_t)i+1)*h+((uint64_t)h+1)*o)*sizeof(float));
	nn->i=i;
	nn->h=h;
	nn->o=o;
	nn->lr=lr;
	nn->hw=(float*)(void*)((uint64_t)(void*)nn+sizeof(struct __NN));
	nn->hb=(float*)(void*)((uint64_t)(void*)nn+sizeof(struct __NN)+i*h*sizeof(float));
	nn->ow=(float*)(void*)((uint64_t)(void*)nn+sizeof(struct __NN)+(i+1)*h*sizeof(float));
	nn->ob=(float*)(void*)((uint64_t)(void*)nn+sizeof(struct __NN)+((i+1)*h+h*o)*sizeof(float));
	for (uint16_t j=0;j<(uint16_t)i*h;j++){
		*(nn->hw+j)=((float)rand())/RAND_MAX*2-1;
	}
	for (uint16_t j=0;j<(uint16_t)h*o;j++){
		*(nn->ow+j)=((float)rand())/RAND_MAX*2-1;
	}
	return nn;
}



void free_neural_network(NN nn){
	free((void*)nn);
}
