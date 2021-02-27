#include <nn.h>
#include <stdlib.h>
#include <stdio.h>



void train(NN nn){
	float tmp[]={0,0,0,0,1,1,1,1,0,1,0,1};
	for (uint8_t i=0;i<12;i+=3){
		neural_network_train(nn,tmp+i,tmp+i+2);
	}
}



int main(int argc,const char** argv){
	NN nn=neural_network(2,2,1,0.1f);
	for (uint32_t i=0;i<1000000;i++){
		train(nn);
	}
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
