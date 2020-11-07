#ifndef __GPU_LSTM_RNN_H__
#define __GPU_LSTM_RNN_H__
#define __LSTM_RNN_ONLY_STRUCT__
#include <lstm-rnn.h>
#include <stdio.h>



#ifdef __cplusplus
extern "C" {
#endif
#ifndef NDEBUG
#define _str2(x) #x
#define _str(x) _str2(x)
#define CUDA_CALL(f) \
	do { \
		cudaError_t __e=f; \
		if (__e!=cudaSuccess){ \
			printf("%s: Line %i (%s): %s\n",__FILE__,__LINE__,__func__,cudaGetErrorString(__e)); \
			raise(SIGABRT); \
		} \
	} while (0)
#define CUDA_GPU_CALL_CUSTOM(f,n,sz,...) \
	do { \
		f<<<n,sz>>>(__VA_ARGS__); \
		cudaError_t __e=cudaGetLastError(); \
		if (__e!=cudaSuccess){ \
			printf("%s: Line %i (%s): %s<<<%lu,%lu>>>(%s): %s\n",__FILE__,__LINE__,__func__,_str(f),n,sz,_str(__VA_ARGS__),cudaGetErrorString(__e)); \
			raise(SIGABRT); \
		} \
		__e=cudaThreadSynchronize(); \
		if (__e!=cudaSuccess){ \
			printf("%s: Line %i (%s): %s<<<%lu,%lu>>>(%s): %s\n",__FILE__,__LINE__,__func__,_str(f),n,sz,_str(__VA_ARGS__),cudaGetErrorString(__e)); \
			raise(SIGABRT); \
		} \
	} while (0)
#define CUDA_GPU_CALL(f,sz,...) CUDA_GPU_CALL_CUSTOM(f,((sz)+BLK_SIZE-1)/BLK_SIZE,BLK_SIZE,__VA_ARGS__)
#else
#define CUDA_CALL(f) (f)
#define CUDA_GPU_CALL_CUSTOM(f,n,sz,...) f<<<n,sz>>>(__VA_ARGS__)
#define CUDA_GPU_CALL(f,sz,...) CUDA_GPU_CALL_CUSTOM(f,((sz)+BLK_SIZE-1)/BLK_SIZE,BLK_SIZE,__VA_ARGS__)
#endif
#define KERNEL_LOOP_IDX_X (blockIdx.x*blockDim.x+threadIdx.x)
#define KERNEL_LOOP_STRIDE_X (blockDim.x*gridDim.x)
#define BLK_SIZE 64
#define DEVICE_ID 0



__declspec(dllexport) void gpu_lstm_rnn_setup_lib(void);



__declspec(dllexport) Dataset gpu_lstm_rnn_create_dataset(float* dt,size_t sz);



__declspec(dllexport) void gpu_lstm_rnn_init_rand(LstmRnn rnn);



__declspec(dllexport) void gpu_lstm_rnn_init_file(LstmRnn rnn,FILE* f);



__declspec(dllexport) void gpu_lstm_rnn_predict_dataset(LstmRnn rnn,Dataset in_,uint32_t ln,float*);



__declspec(dllexport) void gpu_lstm_rnn_predict(LstmRnn rnn,float* in_,uint32_t ln,float*);



__declspec(dllexport) void gpu_lstm_rnn_train_multiple(LstmRnn rnn,Dataset dts,uint8_t e,uint32_t ln,uint32_t s);



__declspec(dllexport) void gpu_lstm_rnn_train(LstmRnn rnn,float* in_,uint32_t ln,float* t);



__declspec(dllexport) void gpu_lstm_rnn_save(LstmRnn rnn,FILE* f);



__declspec(dllexport) void gpu_lstm_rnn_free(LstmRnn rnn);



__declspec(dllexport) void gpu_lstm_rnn_free_dataset(Dataset dt);



#ifdef __cplusplus
}
#endif
#endif
