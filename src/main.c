#include <nn.h>
#include <stdlib.h>
#include <stdio.h>



int main(int argc,const char** argv){
	NN nn=neural_network(2,4,1,0.01f);
	free_neural_network(nn);
	return 0;
}
