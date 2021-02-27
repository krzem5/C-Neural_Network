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
	float* _h;
	float* _e;
};



NN neural_network(uint8_t i,uint8_t h,uint8_t o,float lr);



void neural_network_predict(NN nn,float* in,float* o);



void neural_network_train(NN nn,float* in,float* to);



void neural_network_free(NN nn);



#endif
