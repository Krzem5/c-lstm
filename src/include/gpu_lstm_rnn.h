#ifndef __GPU_LSTM_RNN_H__
#define __GPU_LSTM_RNN_H__
#define __LSTM_RNN_ONLY_STRUCT__
#include <lstm-rnn.h>
#include <stdio.h>
#ifdef __cplusplus
extern "C" {
#endif



__declspec(dllexport) void gpu_lstm_rnn_init_rand(LstmRnn rnn);



__declspec(dllexport) void gpu_lstm_rnn_init_file(LstmRnn rnn,FILE* f);



__declspec(dllexport) float* gpu_lstm_rnn_predict(LstmRnn rnn,float** in_,uint32_t ln);



__declspec(dllexport) void gpu_lstm_rnn_train(LstmRnn rnn,float** in_,uint32_t ln,float** t);



__declspec(dllexport) void gpu_lstm_rnn_save(LstmRnn rnn,FILE* f);



__declspec(dllexport) void gpu_lstm_rnn_free(LstmRnn rnn);



#ifdef __cplusplus
}
#endif
#endif
