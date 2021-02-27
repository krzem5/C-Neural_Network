#ifndef __NN_H__
#define __NN_H__
#include <stdint.h>



typedef struct __NN* NN;



struct __NN{
	uint8_t i;
	uint8_t h;
	uint8_t o;
	float lr;
	float* hw;
	float* hb;
	float* ow;
	float* ob;
};



NN neural_network(uint8_t i,uint8_t h,uint8_t o,float lr);



void free_neural_network(NN nn);



#endif
