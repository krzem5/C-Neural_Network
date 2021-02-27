#include <nn.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <math.h>



#define _SIGMOID(x) (0.5f+(x)/(2*(1+fabsf((x)))))
#define _SIGMOID_DERIVATIVE(x) ((x)/(1-(x)))



NN neural_network(uint8_t i,uint8_t h,uint8_t o,float lr){
	// srand((unsigned int)time(NULL));
	srand((unsigned int)12345);
	NN nn=malloc(sizeof(struct __NN)+(((uint64_t)i+3)*h+((uint64_t)h+1)*o)*sizeof(float));
	nn->i=i;
	nn->h=h;
	nn->o=o;
	nn->lr=lr;
	nn->hw=(float*)(void*)((uint64_t)(void*)nn+sizeof(struct __NN));
	nn->hb=(float*)(void*)((uint64_t)(void*)nn+sizeof(struct __NN)+i*h*sizeof(float));
	nn->ow=(float*)(void*)((uint64_t)(void*)nn+sizeof(struct __NN)+(i+1)*h*sizeof(float));
	nn->ob=(float*)(void*)((uint64_t)(void*)nn+sizeof(struct __NN)+((i+1)*h+h*o)*sizeof(float));
	nn->_h=(float*)(void*)((uint64_t)(void*)nn+sizeof(struct __NN)+((i+1)*h+(h+1)*o)*sizeof(float));
	nn->_e=(float*)(void*)((uint64_t)(void*)nn+sizeof(struct __NN)+((i+2)*h+(h+1)*o)*sizeof(float));
	for (uint16_t j=0;j<(uint16_t)i*h;j++){
		*(nn->hw+j)=((float)rand())/RAND_MAX*2-1;
	}
	for (uint16_t j=0;j<h;j++){
		*(nn->hb+j)=0;
	}
	for (uint16_t j=0;j<(uint16_t)h*o;j++){
		*(nn->ow+j)=((float)rand())/RAND_MAX*2-1;
	}
	for (uint16_t j=0;j<o;j++){
		*(nn->ob+j)=0;
	}
	return nn;
}



void neural_network_predict(NN nn,float* in,float* o){
	for (uint8_t i=0;i<nn->h;i++){
		*(nn->_h+i)=*(nn->hb+i);
		for (uint8_t j=0;j<nn->i;j++){
			*(nn->_h+i)+=*(nn->hw+i*nn->i+j)*(*(in+j));
		}
		*(nn->_h+i)=_SIGMOID(*(nn->_h+i));
	}
	for (uint8_t i=0;i<nn->o;i++){
		*(o+i)=*(nn->ob+i);
		for (uint8_t j=0;j<nn->h;j++){
			*(o+i)+=*(nn->ow+i*nn->h+j)*(*(nn->_h+j));
		}
		*(o+i)=_SIGMOID(*(o+i));
	}
}



void neural_network_train(NN nn,float* in,float* to){
	for (uint8_t i=0;i<nn->h;i++){
		*(nn->_h+i)=*(nn->hb+i);
		for (uint8_t j=0;j<nn->i;j++){
			*(nn->_h+i)+=*(nn->hw+i*nn->i+j)*(*(in+j));
		}
		*(nn->_h+i)=_SIGMOID(*(nn->_h+i));
		*(nn->_e+i)=0;
	}
	for (uint8_t i=0;i<nn->o;i++){
		float v=*(nn->ob+i);
		for (uint8_t j=0;j<nn->h;j++){
			v+=*(nn->ow+i*nn->h+j)*(*(nn->_h+j));
		}
		v=_SIGMOID(v);
		float oe=*(to+i)-v;
		float g=_SIGMOID_DERIVATIVE(v)*oe*nn->lr;
		*(nn->ob+i)+=g;
		for (uint8_t j=0;j<nn->h;j++){
			*(nn->ow+i*nn->h+j)+=*(nn->_h+j)*v;
			*(nn->_e+j)+=*(nn->ow+i*nn->h+j)*oe;
		}
	}
	for (uint8_t i=0;i<nn->h;i++){
		float g=_SIGMOID_DERIVATIVE(*(nn->_h+i))*(*(nn->_e+i))*nn->lr;
		*(nn->hb+i)+=g;
		for (uint8_t j=0;j<nn->i;j++){
			*(nn->hw+i*nn->i+j)+=*(in+j)*g;
		}
	}
}



void neural_network_free(NN nn){
	free((void*)nn);
}
