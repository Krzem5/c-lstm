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



int WINAPI DllMain(void* dll,unsigned long r,void* rs){
	(void)dll;
	(void)r;
	(void)rs;
	return TRUE;
}
