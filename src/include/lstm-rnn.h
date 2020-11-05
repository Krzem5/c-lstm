#ifndef __LSTM_RNN_H__
#define __LSTM_RNN_H__
#include <stdint.h>
#include <limits.h>
#include <signal.h>



#ifndef __cplusplus
#ifdef NULL
#undef NULL
#endif
#define NULL ((void*)0)
#define bool _Bool
#define true 1
#define false 0
#define assert(ex) \
	do{ \
		if (!(ex)){ \
			printf("%s:%d (%s): Assertion Failed: %s\n",__FILE__,__LINE__,__func__,#ex); \
			raise(SIGABRT); \
		} \
	} while (0)
#else
extern "C" {
#endif



typedef struct __LSTMRNN* LstmRnn;
typedef float* Dataset;



enum RNN_BACKEND{
	RNN_BACKEND_CPU=0x00,
	RNN_BACKEND_GPU=0x01
};



struct __LSTMRNN_LSTM_LAYER{
	uint8_t x;
	uint8_t y;
	float* wx;
	float* wf;
	float* wi;
	float* wo;
	float* bx;
	float* bf;
	float* bi;
	float* bo;
	uint16_t _xy;
	float* _cl;
	float* _xhl;
	float* _cal;
	float* _fl;
	float* _il;
	float* _ol;
	float* _outl;
	float* _c;
	float* _h;
	float* _hg;
	float* _cg;
	float* _wxg;
	float* _wfg;
	float* _wig;
	float* _wog;
	float* _bxg;
	float* _bfg;
	float* _big;
	float* _bog;
};



struct __LSTMRNN_LSTM_LAYER2{
	uint8_t x;
	uint8_t y;
	float* wx;
	float* wf;
	float* wi;
	float* wo;
	float* bx;
	float* bf;
	float* bi;
	float* bo;
	uint16_t _xy;
	uint32_t _sz;
	float** _cl;
	float** _xhl;
	float** _cal;
	float** _fl;
	float** _il;
	float** _ol;
	float** _outl;
	float* _c;
	float* _h;
	float* _hg;
	float* _cg;
	float* _wxg;
	float* _wfg;
	float* _wig;
	float* _wog;
	float* _bxg;
	float* _bfg;
	float* _big;
	float* _bog;
};



struct __LSTMRNN_FULLY_CONNECTED_LAYER{
	uint8_t x;
	uint8_t y;
	float* w;
	float* b;
};



struct __LSTMRNN{
	const char* fp;
	uint8_t i;
	uint8_t h;
	uint8_t o;
	float lr;
	union{
		struct _cpu{
			struct __LSTMRNN_LSTM_LAYER2* lstm;
			struct __LSTMRNN_FULLY_CONNECTED_LAYER* fc;
		} cpu;
		struct _gpu{
			struct __LSTMRNN_LSTM_LAYER* lstm;
			struct __LSTMRNN_FULLY_CONNECTED_LAYER* fc;
			struct __LSTMRNN_LSTM_LAYER* lstm_d;
			struct __LSTMRNN_FULLY_CONNECTED_LAYER* fc_d;
			float* cfio;
			float* to;
		} gpu;
	} dt;
};



#ifndef __LSTM_RNN_ONLY_STRUCT__



bool set_rnn_backend(enum RNN_BACKEND t);



Dataset create_dataset(float* dt,size_t sz);



LstmRnn init_lstm_rnn(const char* fp,uint8_t i,uint8_t h,uint8_t o,float lr);



float* lstm_rnn_predict_dataset(LstmRnn rnn,Dataset in_,uint32_t ln);



float* lstm_rnn_predict(LstmRnn rnn,float* in_,uint32_t ln);



void lstm_rnn_train_multiple(LstmRnn rnn,Dataset dts,uint8_t e,uint32_t ln,uint32_t s);



void lstm_rnn_train(LstmRnn rnn,float* in_,uint32_t ln,float* t);



void save_lstm_rnn(LstmRnn rnn);



void free_lstm_rnn(LstmRnn rnn);



void free_dataset(Dataset dts);



#endif
#ifdef __cplusplus
}
#endif
#endif
