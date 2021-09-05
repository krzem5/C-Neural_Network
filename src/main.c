#include <nn.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>



void train(NN nn){
	float dt[]={0,0,0,0,1,1,1,1,0,1,0,1};
	for (uint8_t i=0;i<12;i+=3){
		neural_network_train(nn,dt+i,dt+i+2);
	}
}



void acc(NN nn){
	float dt[]={0,0,0,0,1,1,1,1,0,1,0,1};
	float* tmp=malloc(sizeof(float));
	float a=0;
	for (uint8_t i=0;i<12;i+=3){
		neural_network_predict(nn,dt+i,tmp);
		a+=fabsf(*(dt+i+2)-(*tmp));
	}
	free(tmp);
	printf("Acc: %.2f%%\n",100-a/4*100);
}



int main(int argc,const char** argv){
	NN nn=neural_network(2,3,1,0.1f);
	acc(nn);
	for (uint32_t i=0;i<10000;i++){
		train(nn);
	}
	acc(nn);
	float* tmp=malloc(3*sizeof(float));
	*tmp=0;
	*(tmp+1)=0;
	neural_network_predict(nn,tmp,tmp+2);
	printf("0 ^ 0 => %f\n",*(tmp+2));
	*(tmp+1)=1;
	neural_network_predict(nn,tmp,tmp+2);
	printf("0 ^ 1 => %f\n",*(tmp+2));
	*tmp=1;
	neural_network_predict(nn,tmp,tmp+2);
	printf("1 ^ 1 => %f\n",*(tmp+2));
	*(tmp+1)=0;
	neural_network_predict(nn,tmp,tmp+2);
	printf("1 ^ 0 => %f\n",*(tmp+2));
	free(tmp);
	neural_network_free(nn);
	return 0;
}
