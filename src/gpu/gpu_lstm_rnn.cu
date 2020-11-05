#include <gpu_lstm_rnn.h>
#include <windows.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>



void gpu_lstm_rnn_init_rand(LstmRnn rnn){
	/**/
}



void gpu_lstm_rnn_init_file(LstmRnn rnn,FILE* f){
	/**/
}



float* gpu_lstm_rnn_predict(LstmRnn rnn,float** in_,uint32_t ln){
	/**/
	return NULL;
}



void gpu_lstm_rnn_train(LstmRnn rnn,float** in_,uint32_t ln,float** t){
	/**/
}



void gpu_lstm_rnn_save(LstmRnn rnn,FILE* f){
	/**/
}



void gpu_lstm_rnn_free(LstmRnn rnn){
	/**/
}



// #define PARALLEL 1



// #ifdef PARALLEL
// __global__ void add(int n,float* x,float* y){
// 	int index=blockIdx.x*blockDim.x+threadIdx.x;
// 	int stride=blockDim.x*gridDim.x;
// 	for (int i=index;i<n;i+=stride){
// 		y[i]=x[i]+y[i];
// 	}
// }
// #else
// void add(int n,float* x,float* y){
// 	for (int i=0;i<n;i++){
// 		y[i]=x[i]+y[i];
// 	}
// }
// #endif



// void hello(const char* s){
// 	printf("Hello %s\n",s);
// 	printf("A!\n");
// 	float* x;
// 	float* y;
// 	int N=1<<30;
// 	printf("B!\n");
// 	cudaMallocManaged(&x,N*sizeof(float));
// 	cudaMallocManaged(&y,N*sizeof(float));
// 	printf("C!\n");
// 	// for (int i=0;i<N;i++){
// 	// 	x[i]=1.0f;
// 	// 	y[i]=2.0f;
// 	// }
// 	printf("START!\n");
// #ifdef PARALLEL
// 	int blockSize=256;
// 	add<<<(N+blockSize-1)/blockSize,blockSize>>>(N,x,y);
// 	cudaDeviceSynchronize();
// #else
// 	add(N,x,y);
// #endif
// 	printf("END!\n");
// 	cudaFree(x);
// 	printf("A!\n");
// 	cudaFree(y);
// 	printf("B!\n");
// }



int WINAPI DllMain(void* dll,unsigned long r,void* rs){
	(void)dll;
	(void)r;
	(void)rs;
	return TRUE;
}
