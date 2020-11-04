#ifndef __LSTM_RNN_H__
#define __LSTM_RNN_H__
#include <stdint.h>
#include <limits.h>
#include <signal.h>



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



typedef struct __LSTMRNN* LstmRnn;



struct __LSTMRNN_LSTM_LAYER{
	uint8_t x;
	uint8_t y;
	float** wx;
	float** wf;
	float** wi;
	float** wo;
	float* bx;
	float* bf;
	float* bi;
	float* bo;
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
	float** _wxg;
	float** _wfg;
	float** _wig;
	float** _wog;
	float* _bxg;
	float* _bfg;
	float* _big;
	float* _bog;
};



struct __LSTMRNN_FULLY_CONNECTED_LAYER{
	uint8_t x;
	uint8_t y;
	float** w;
	float* b;
};



struct __LSTMRNN{
	const char* fp;
	uint8_t i;
	uint8_t h;
	uint8_t o;
	float lr;
	struct __LSTMRNN_LSTM_LAYER* lstm;
	struct __LSTMRNN_FULLY_CONNECTED_LAYER* fc;
};



LstmRnn init_lstm_rnn(const char* fp,uint8_t i,uint8_t h,uint8_t o,float lr);



float* lstm_rnn_predict(LstmRnn rnn,float** dt,uint32_t ln);



void lstm_rnn_train(LstmRnn rnn,float** in_,uint32_t ln,float** t);



void save_lstm_rnn(LstmRnn rnn);



void free_lstm_rnn(LstmRnn rnn);



#endif
