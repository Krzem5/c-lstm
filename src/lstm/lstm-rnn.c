#include <lstm-rnn.h>
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>



typedef void (*_gpu_lib_setup_t)(void);
typedef Dataset (*_gpu_lib_create_dataset_t)(float* dt,size_t sz);
typedef void (*_gpu_lib_init_rand_t)(LstmRnn rnn);
typedef void (*_gpu_lib_init_file_t)(LstmRnn rnn,FILE* f);
typedef void (*_gpu_lib_predict_dataset_t)(LstmRnn rnn,Dataset in_,uint32_t ln,float* o);
typedef void (*_gpu_lib_predict_t)(LstmRnn rnn,float* in_,uint32_t ln,float* o);
typedef void (*_gpu_lib_train_multiple_t)(LstmRnn rnn,Dataset dts,uint8_t e,uint32_t ln,uint32_t s);
typedef void (*_gpu_lib_train_t)(LstmRnn rnn,float* in_,uint32_t ln,float* t);
typedef void (*_gpu_lib_save_t)(LstmRnn rnn,FILE* f);
typedef void (*_gpu_lib_free_t)(LstmRnn rnn);
typedef void (*_gpu_lib_free_dataset_t)(Dataset dts);
enum RNN_BACKEND _bt=RNN_BACKEND_CPU;
HMODULE _gpu_lib=NULL;
_gpu_lib_setup_t _gpu_lib_setup=NULL;
_gpu_lib_create_dataset_t _gpu_lib_create_dataset=NULL;
_gpu_lib_init_rand_t _gpu_lib_init_rand=NULL;
_gpu_lib_init_file_t _gpu_lib_init_file=NULL;
_gpu_lib_predict_dataset_t _gpu_lib_predict_dataset=NULL;
_gpu_lib_predict_t _gpu_lib_predict=NULL;
_gpu_lib_train_multiple_t _gpu_lib_train_multiple=NULL;
_gpu_lib_train_t _gpu_lib_train=NULL;
_gpu_lib_save_t _gpu_lib_save=NULL;
_gpu_lib_free_t _gpu_lib_free=NULL;
_gpu_lib_free_dataset_t _gpu_lib_free_dataset=NULL;



void _free_gpu_lib(void){
	if (_gpu_lib!=NULL){
		FreeLibrary(_gpu_lib);
		_gpu_lib=NULL;
	}
}



float sigmoidf(float x){
	return 1.0f/(1+expf(-x));
}
float tanh_d(float x){
	return 1-x*x;
}
float sigmoid_d(float x){
	return (1-x)*x;
}



float* _lstm_fwd(struct __LSTMRNN_LSTM_LAYER_CPU* lstm,float* in_){
	for (uint8_t i=0;i<lstm->y;i++){
		float ca=lstm->bx[i];
		float f=lstm->bf[i];
		float i_=lstm->bi[i];
		float o=lstm->bo[i];
		for (uint16_t j=0;j<lstm->_xy;j++){
			float xh=(j<lstm->x?in_[j]:lstm->_h[j-lstm->x]);
			ca+=lstm->wx[i*lstm->_xy+j]*xh;
			f+=lstm->wf[i*lstm->_xy+j]*xh;
			i_+=lstm->wi[i*lstm->_xy+j]*xh;
			o+=lstm->wo[i*lstm->_xy+j]*xh;
		}
		lstm->_c[i]=tanhf(ca)*sigmoidf(i_)+lstm->_c[i]*sigmoidf(f);
		lstm->_h[i]=tanhf(lstm->_c[i])*sigmoidf(o);
	}
	return lstm->_h;
}



float* _lstm_fwd_t(struct __LSTMRNN_LSTM_LAYER_CPU* lstm,float* in_){
	if (lstm->_sz==-1){
		lstm->_hg=malloc(lstm->y*sizeof(float));
		lstm->_cg=malloc(lstm->y*sizeof(float));
		lstm->_wxg=malloc(lstm->y*lstm->_xy*sizeof(float));
		lstm->_wfg=malloc(lstm->y*lstm->_xy*sizeof(float));
		lstm->_wig=malloc(lstm->y*lstm->_xy*sizeof(float));
		lstm->_wog=malloc(lstm->y*lstm->_xy*sizeof(float));
		lstm->_bxg=malloc(lstm->y*sizeof(float));
		lstm->_bfg=malloc(lstm->y*sizeof(float));
		lstm->_big=malloc(lstm->y*sizeof(float));
		lstm->_bog=malloc(lstm->y*sizeof(float));
	}
	float* lc=malloc(lstm->y*sizeof(float));
	float* xh=malloc(lstm->_xy*sizeof(float));
	float* ca=malloc(lstm->y*sizeof(float));
	float* f=malloc(lstm->y*sizeof(float));
	float* i_=malloc(lstm->y*sizeof(float));
	float* o=malloc(lstm->y*sizeof(float));
	float* out=malloc(lstm->y*sizeof(float));
	for (uint8_t i=0;i<lstm->y;i++){
		if (lstm->_sz==-1){
			lstm->_hg[i]=0;
			lstm->_cg[i]=0;
			lstm->_bxg[i]=0;
			lstm->_bfg[i]=0;
			lstm->_big[i]=0;
			lstm->_bog[i]=0;
		}
		ca[i]=lstm->bx[i];
		f[i]=lstm->bf[i];
		i_[i]=lstm->bi[i];
		o[i]=lstm->bo[i];
		for (uint16_t j=0;j<lstm->_xy;j++){
			if (lstm->_sz==-1){
				if (i==lstm->y-1&&j==lstm->_xy-1){
					lstm->_sz=0;
				}
				lstm->_wxg[i*lstm->_xy+j]=0;
				lstm->_wfg[i*lstm->_xy+j]=0;
				lstm->_wig[i*lstm->_xy+j]=0;
				lstm->_wog[i*lstm->_xy+j]=0;
			}
			if (i==0){
				if (j<lstm->x){
					xh[j]=in_[j];
				}
				else{
					xh[j]=lstm->_h[j-lstm->x];
					lc[j-lstm->x]=lstm->_c[j-lstm->x];
				}
			}
			ca[i]+=lstm->wx[i*lstm->_xy+j]*xh[j];
			f[i]+=lstm->wf[i*lstm->_xy+j]*xh[j];
			i_[i]+=lstm->wi[i*lstm->_xy+j]*xh[j];
			o[i]+=lstm->wo[i*lstm->_xy+j]*xh[j];
		}
		ca[i]=tanhf(ca[i]);
		f[i]=sigmoidf(f[i]);
		i_[i]=sigmoidf(i_[i]);
		o[i]=sigmoidf(o[i]);
		lstm->_c[i]=ca[i]*i_[i]+lstm->_c[i]*f[i];
		out[i]=tanhf(lstm->_c[i]);
		lstm->_h[i]=out[i]*o[i];
	}
	lstm->_sz++;
	lstm->_cl=realloc(lstm->_cl,lstm->_sz*sizeof(float*));
	lstm->_xhl=realloc(lstm->_xhl,lstm->_sz*sizeof(float*));
	lstm->_cal=realloc(lstm->_cal,lstm->_sz*sizeof(float*));
	lstm->_fl=realloc(lstm->_fl,lstm->_sz*sizeof(float*));
	lstm->_il=realloc(lstm->_il,lstm->_sz*sizeof(float*));
	lstm->_ol=realloc(lstm->_ol,lstm->_sz*sizeof(float*));
	lstm->_outl=realloc(lstm->_outl,lstm->_sz*sizeof(float*));
	lstm->_cl[lstm->_sz-1]=lc;
	lstm->_xhl[lstm->_sz-1]=xh;
	lstm->_cal[lstm->_sz-1]=ca;
	lstm->_fl[lstm->_sz-1]=f;
	lstm->_il[lstm->_sz-1]=i_;
	lstm->_ol[lstm->_sz-1]=o;
	lstm->_outl[lstm->_sz-1]=out;
	return lstm->_h;
}



void _lstm_train(struct __LSTMRNN_LSTM_LAYER_CPU* lstm,float* tg){
	lstm->_sz--;
	float* c=lstm->_cl[lstm->_sz];
	float* xh=lstm->_xhl[lstm->_sz];
	float* ca=lstm->_cal[lstm->_sz];
	float* f=lstm->_fl[lstm->_sz];
	float* i_=lstm->_il[lstm->_sz];
	float* o=lstm->_ol[lstm->_sz];
	float* out=lstm->_outl[lstm->_sz];
	tg[0]+=lstm->_hg[0];
	for (uint8_t i=0;i<lstm->y;i++){
		lstm->_cg[i]=tanh_d(out[i])*o[i]*tg[i]+lstm->_cg[i];
		float lfg=c[i]*lstm->_cg[i]*sigmoid_d(f[i]);
		lstm->_cg[i]*=f[i];
		float lxg=tanh_d(ca[i])*i_[i]*lstm->_cg[i];
		float lig=ca[i]*lstm->_cg[i]*sigmoid_d(i_[i]);
		float log=out[i]*tg[i]*sigmoid_d(o[i]);
		lstm->_bxg[i]+=lxg;
		lstm->_big[i]+=lig;
		lstm->_bfg[i]+=lfg;
		lstm->_bog[i]+=log;
		for (uint8_t j=0;j<lstm->_xy;j++){
			if (j>=lstm->x){
				if (i==0){
					if (j>lstm->x){
						tg[j-lstm->x]+=lstm->_hg[j-lstm->x];
					}
					lstm->_hg[j-lstm->x]=0;
				}
				lstm->_hg[j-lstm->x]+=lstm->wx[i*lstm->_xy+j]*lxg+lstm->wi[i*lstm->_xy+j]*lig+lstm->wf[i*lstm->_xy+j]*lfg+lstm->wo[i*lstm->_xy+j]*log;
			}
			lstm->_wxg[i*lstm->_xy+j]+=lxg*xh[j];
			lstm->_wig[i*lstm->_xy+j]+=lig*xh[j];
			lstm->_wfg[i*lstm->_xy+j]+=lfg*xh[j];
			lstm->_wog[i*lstm->_xy+j]+=log*xh[j];
		}
	}
	free(c);
	free(xh);
	free(ca);
	free(f);
	free(i_);
	free(o);
	free(out);
}



void _lstm_update(struct __LSTMRNN_LSTM_LAYER_CPU* lstm,float lr){
	lstm->_sz=0;
	free(lstm->_cl);
	free(lstm->_xhl);
	free(lstm->_cal);
	free(lstm->_fl);
	free(lstm->_il);
	free(lstm->_ol);
	free(lstm->_outl);
	lstm->_cl=NULL;
	lstm->_xhl=NULL;
	lstm->_cal=NULL;
	lstm->_fl=NULL;
	lstm->_il=NULL;
	lstm->_ol=NULL;
	lstm->_outl=NULL;
	for (uint8_t i=0;i<lstm->y;i++){
		lstm->_c[i]=0;
		lstm->_h[i]=0;
		lstm->_hg[i]=0;
		lstm->_cg[i]=0;
		for (uint16_t j=0;j<lstm->_xy;j++){
			lstm->wx[i*lstm->_xy+j]-=lstm->_wxg[i*lstm->_xy+j]*lr;
			lstm->wf[i*lstm->_xy+j]-=lstm->_wfg[i*lstm->_xy+j]*lr;
			lstm->wi[i*lstm->_xy+j]-=lstm->_wig[i*lstm->_xy+j]*lr;
			lstm->wo[i*lstm->_xy+j]-=lstm->_wog[i*lstm->_xy+j]*lr;
			lstm->_wxg[i*lstm->_xy+j]=0;
			lstm->_wfg[i*lstm->_xy+j]=0;
			lstm->_wig[i*lstm->_xy+j]=0;
			lstm->_wog[i*lstm->_xy+j]=0;
		}
		lstm->bx[i]-=lstm->_bxg[i]*lr;
		lstm->bf[i]-=lstm->_bfg[i]*lr;
		lstm->bi[i]-=lstm->_big[i]*lr;
		lstm->bo[i]-=lstm->_bog[i]*lr;
		lstm->_bxg[i]=0;
		lstm->_bfg[i]=0;
		lstm->_big[i]=0;
		lstm->_bog[i]=0;
	}
}



void _lstm_reset(struct __LSTMRNN_LSTM_LAYER_CPU* lstm){
	for (uint8_t i=0;i<lstm->y;i++){
		lstm->_c[i]=0;
		lstm->_h[i]=0;
	}
}



float* _fc_fwd(struct __LSTMRNN_FULLY_CONNECTED_LAYER* fc,float* in_){
	float* o=malloc(fc->y*sizeof(float));
	for (uint8_t i=0;i<fc->y;i++){
		o[i]=fc->b[i];
		for (uint8_t j=0;j<fc->x;j++){
			o[i]+=fc->w[i*fc->x+j]*in_[j];
		}
	}
	return o;
}



float* _fc_train(struct __LSTMRNN_FULLY_CONNECTED_LAYER* fc,float* in_,float* tg,float lr){
	float* o=malloc(fc->x*sizeof(float));
	for (uint8_t i=0;i<fc->y;i++){
		float p=fc->b[i];
		for (uint8_t j=0;j<fc->x;j++){
			p+=fc->w[i*fc->x+j]*in_[j];
		}
		float bg=p-tg[i];
		fc->b[i]-=bg*lr;
		for (uint8_t j=0;j<fc->x;j++){
			if (i==0){
				o[j]=0;
			}
			o[j]+=fc->w[i*fc->x+j]*bg;
			fc->w[i*fc->x+j]-=bg*in_[j]*lr;
		}
	}
	return o;
}



bool set_rnn_backend(enum RNN_BACKEND t){
	_bt=t;
	if (t==RNN_BACKEND_GPU&&_gpu_lib==NULL){
		_gpu_lib=LoadLibraryA("gpu_kernel.dll");
		if (_gpu_lib==NULL){
			printf("ERROR: DLL 'gpu_kernel.dll' not Found.\n");
			return false;
		}
		atexit(_free_gpu_lib);
		_gpu_lib_setup=(_gpu_lib_setup_t)GetProcAddress(_gpu_lib,"gpu_lstm_rnn_setup_lib");
		if (_gpu_lib_setup==NULL){
			printf("ERROR: Function 'gpu_lstm_rnn_setup_lib' not Found in the DLL.\n");
			return false;
		}
		_gpu_lib_setup();
		_gpu_lib_create_dataset=(_gpu_lib_create_dataset_t)GetProcAddress(_gpu_lib,"gpu_lstm_rnn_create_dataset");
		if (_gpu_lib_create_dataset==NULL){
			printf("ERROR: Function 'gpu_lstm_rnn_create_dataset' not Found in the DLL.\n");
			return false;
		}
		_gpu_lib_init_rand=(_gpu_lib_init_rand_t)GetProcAddress(_gpu_lib,"gpu_lstm_rnn_init_rand");
		if (_gpu_lib_init_rand==NULL){
			printf("ERROR: Function 'gpu_lstm_rnn_init_rand' not Found in the DLL.\n");
			return false;
		}
		_gpu_lib_init_file=(_gpu_lib_init_file_t)GetProcAddress(_gpu_lib,"gpu_lstm_rnn_init_file");
		if (_gpu_lib_init_file==NULL){
			printf("ERROR: Function 'gpu_lstm_rnn_init_file' not Found in the DLL.\n");
			return false;
		}
		_gpu_lib_predict_dataset=(_gpu_lib_predict_dataset_t)GetProcAddress(_gpu_lib,"gpu_lstm_rnn_predict_dataset");
		if (_gpu_lib_predict_dataset==NULL){
			printf("ERROR: Function 'gpu_lstm_rnn_predict_dataset' not Found in the DLL.\n");
			return false;
		}
		_gpu_lib_predict=(_gpu_lib_predict_t)GetProcAddress(_gpu_lib,"gpu_lstm_rnn_predict");
		if (_gpu_lib_predict==NULL){
			printf("ERROR: Function 'gpu_lstm_rnn_predict' not Found in the DLL.\n");
			return false;
		}
		_gpu_lib_train_multiple=(_gpu_lib_train_multiple_t)GetProcAddress(_gpu_lib,"gpu_lstm_rnn_train_multiple");
		if (_gpu_lib_train_multiple==NULL){
			printf("ERROR: Function 'gpu_lstm_rnn_train_multiple' not Found in the DLL.\n");
			return false;
		}
		_gpu_lib_train=(_gpu_lib_train_t)GetProcAddress(_gpu_lib,"gpu_lstm_rnn_train");
		if (_gpu_lib_train==NULL){
			printf("ERROR: Function 'gpu_lstm_rnn_train' not Found in the DLL.\n");
			return false;
		}
		_gpu_lib_save=(_gpu_lib_save_t)GetProcAddress(_gpu_lib,"gpu_lstm_rnn_save");
		if (_gpu_lib_save==NULL){
			printf("ERROR: Function 'gpu_lstm_rnn_save' not Found in the DLL.\n");
			return false;
		}
		_gpu_lib_free=(_gpu_lib_free_t)GetProcAddress(_gpu_lib,"gpu_lstm_rnn_free");
		if (_gpu_lib_free==NULL){
			printf("ERROR: Function 'gpu_lstm_rnn_free' not Found in the DLL.\n");
			return false;
		}
		_gpu_lib_free_dataset=(_gpu_lib_free_dataset_t)GetProcAddress(_gpu_lib,"gpu_lstm_rnn_free_dataset");
		if (_gpu_lib_free_dataset==NULL){
			printf("ERROR: Function 'gpu_lstm_rnn_free_dataset' not Found in the DLL.\n");
			return false;
		}
	}
	return true;
}



Dataset create_dataset(float* dt,size_t sz){
	if (_bt==RNN_BACKEND_GPU){
		return _gpu_lib_create_dataset(dt,sz);
	}
	return dt;
}



LstmRnn init_lstm_rnn(const char* fp,uint8_t in,uint8_t hn,uint8_t on,float lr){
	LstmRnn o=malloc(sizeof(struct __LSTMRNN));
	o->fp=fp;
	o->i=in;
	o->h=hn;
	o->o=on;
	o->lr=lr;
	if (_bt==RNN_BACKEND_CPU){
		o->dt.cpu.lstm=malloc(sizeof(struct __LSTMRNN_LSTM_LAYER_CPU));
		o->dt.cpu.lstm->x=in;
		o->dt.cpu.lstm->y=hn;
		o->dt.cpu.lstm->_xy=o->dt.cpu.lstm->x+o->dt.cpu.lstm->y;
		o->dt.cpu.lstm->wx=malloc(o->dt.cpu.lstm->y*o->dt.cpu.lstm->_xy*sizeof(float));
		o->dt.cpu.lstm->wf=malloc(o->dt.cpu.lstm->y*o->dt.cpu.lstm->_xy*sizeof(float));
		o->dt.cpu.lstm->wi=malloc(o->dt.cpu.lstm->y*o->dt.cpu.lstm->_xy*sizeof(float));
		o->dt.cpu.lstm->wo=malloc(o->dt.cpu.lstm->y*o->dt.cpu.lstm->_xy*sizeof(float));
		o->dt.cpu.lstm->bx=malloc(o->dt.cpu.lstm->y*sizeof(float));
		o->dt.cpu.lstm->bf=malloc(o->dt.cpu.lstm->y*sizeof(float));
		o->dt.cpu.lstm->bi=malloc(o->dt.cpu.lstm->y*sizeof(float));
		o->dt.cpu.lstm->bo=malloc(o->dt.cpu.lstm->y*sizeof(float));
		o->dt.cpu.lstm->_sz=-1;
		o->dt.cpu.lstm->_cl=NULL;
		o->dt.cpu.lstm->_xhl=NULL;
		o->dt.cpu.lstm->_cal=NULL;
		o->dt.cpu.lstm->_fl=NULL;
		o->dt.cpu.lstm->_il=NULL;
		o->dt.cpu.lstm->_ol=NULL;
		o->dt.cpu.lstm->_outl=NULL;
		o->dt.cpu.lstm->_c=malloc(o->dt.cpu.lstm->y*sizeof(float));
		o->dt.cpu.lstm->_h=malloc(o->dt.cpu.lstm->y*sizeof(float));
		o->dt.cpu.lstm->_hg=NULL;
		o->dt.cpu.lstm->_cg=NULL;
		o->dt.cpu.lstm->_wxg=NULL;
		o->dt.cpu.lstm->_wfg=NULL;
		o->dt.cpu.lstm->_wig=NULL;
		o->dt.cpu.lstm->_wog=NULL;
		o->dt.cpu.lstm->_bxg=NULL;
		o->dt.cpu.lstm->_bfg=NULL;
		o->dt.cpu.lstm->_big=NULL;
		o->dt.cpu.lstm->_bog=NULL;
		o->dt.cpu.fc=malloc(sizeof(struct __LSTMRNN_FULLY_CONNECTED_LAYER));
		o->dt.cpu.fc->x=hn;
		o->dt.cpu.fc->y=on;
		o->dt.cpu.fc->w=malloc(o->dt.cpu.fc->y*o->dt.cpu.fc->x*sizeof(float));
		o->dt.cpu.fc->b=malloc(o->dt.cpu.fc->y*sizeof(float));
	}
	if (GetFileAttributesA(o->fp)==INVALID_FILE_ATTRIBUTES&&GetLastError()==ERROR_FILE_NOT_FOUND){
		if (_bt==RNN_BACKEND_GPU){
			_gpu_lib_init_rand(o);
		}
		else{
			for (uint8_t i=0;i<o->dt.cpu.lstm->y;i++){
				o->dt.cpu.lstm->_c[i]=0;
				o->dt.cpu.lstm->_h[i]=0;
				for (uint16_t j=0;j<o->dt.cpu.lstm->_xy;j++){
					o->dt.cpu.lstm->wx[i*o->dt.cpu.lstm->_xy+j]=((float)rand())/RAND_MAX*0.2f-0.1f;
					o->dt.cpu.lstm->wf[i*o->dt.cpu.lstm->_xy+j]=((float)rand())/RAND_MAX*0.2f-0.1f;
					o->dt.cpu.lstm->wi[i*o->dt.cpu.lstm->_xy+j]=((float)rand())/RAND_MAX*0.2f-0.1f;
					o->dt.cpu.lstm->wo[i*o->dt.cpu.lstm->_xy+j]=((float)rand())/RAND_MAX*0.2f-0.1f;
				}
				o->dt.cpu.lstm->bx[i]=((float)rand())/RAND_MAX*0.2f-0.1f;
				o->dt.cpu.lstm->bf[i]=((float)rand())/RAND_MAX*0.2f-0.1f;
				o->dt.cpu.lstm->bi[i]=((float)rand())/RAND_MAX*0.2f-0.1f;
				o->dt.cpu.lstm->bo[i]=((float)rand())/RAND_MAX*0.2f-0.1f;
			}
			for (uint8_t i=0;i<o->dt.cpu.fc->y;i++){
				for (uint8_t j=0;j<o->dt.cpu.fc->x;j++){
					o->dt.cpu.fc->w[i*o->dt.cpu.fc->x+j]=((float)rand())/RAND_MAX*2-1;
				}
				o->dt.cpu.fc->b[i]=0;
			}
		}
	}
	else{
		FILE* f=NULL;
		assert(fopen_s(&f,o->fp,"rb")==0);
		if (_bt==RNN_BACKEND_GPU){
			_gpu_lib_init_file(o,f);
		}
		else{
			assert(fread((void*)o->dt.cpu.lstm->bx,sizeof(float),o->dt.cpu.lstm->y,f)==o->dt.cpu.lstm->y);
			assert(fread((void*)o->dt.cpu.lstm->bf,sizeof(float),o->dt.cpu.lstm->y,f)==o->dt.cpu.lstm->y);
			assert(fread((void*)o->dt.cpu.lstm->bi,sizeof(float),o->dt.cpu.lstm->y,f)==o->dt.cpu.lstm->y);
			assert(fread((void*)o->dt.cpu.lstm->bo,sizeof(float),o->dt.cpu.lstm->y,f)==o->dt.cpu.lstm->y);
			assert(fread((void*)o->dt.cpu.lstm->wx,sizeof(float),o->dt.cpu.lstm->y*o->dt.cpu.lstm->_xy,f)==o->dt.cpu.lstm->y*o->dt.cpu.lstm->_xy);
			assert(fread((void*)o->dt.cpu.lstm->wf,sizeof(float),o->dt.cpu.lstm->y*o->dt.cpu.lstm->_xy,f)==o->dt.cpu.lstm->y*o->dt.cpu.lstm->_xy);
			assert(fread((void*)o->dt.cpu.lstm->wi,sizeof(float),o->dt.cpu.lstm->y*o->dt.cpu.lstm->_xy,f)==o->dt.cpu.lstm->y*o->dt.cpu.lstm->_xy);
			assert(fread((void*)o->dt.cpu.lstm->wo,sizeof(float),o->dt.cpu.lstm->y*o->dt.cpu.lstm->_xy,f)==o->dt.cpu.lstm->y*o->dt.cpu.lstm->_xy);
			assert(fread((void*)o->dt.cpu.fc->b,sizeof(float),o->dt.cpu.fc->y,f)==o->dt.cpu.fc->y);
			assert(fread((void*)o->dt.cpu.fc->w,sizeof(float),o->dt.cpu.fc->y*o->dt.cpu.fc->x,f)==o->dt.cpu.fc->y*o->dt.cpu.fc->x);
		}
		for (uint8_t i=0;i<o->dt.cpu.lstm->y;i++){
			o->dt.cpu.lstm->_c[i]=0;
			o->dt.cpu.lstm->_h[i]=0;
		}
		fclose(f);
	}
	return o;
}



float* lstm_rnn_predict_dataset(LstmRnn rnn,Dataset in_,uint32_t ln){
	if (_bt==RNN_BACKEND_GPU){
		float* o=malloc(rnn->o*sizeof(float));
		_gpu_lib_predict_dataset(rnn,in_,ln,o);
		return o;
	}
	for (uint32_t i=0;i<ln-1;i++){
		_lstm_fwd(rnn->dt.cpu.lstm,in_+i*rnn->i);
	}
	float* o=_fc_fwd(rnn->dt.cpu.fc,_lstm_fwd(rnn->dt.cpu.lstm,in_+(ln-1)*rnn->i));
	_lstm_reset(rnn->dt.cpu.lstm);
	return o;
}



float* lstm_rnn_predict(LstmRnn rnn,float* in_,uint32_t ln){
	if (_bt==RNN_BACKEND_GPU){
		float* o=malloc(rnn->o*sizeof(float));
		_gpu_lib_predict(rnn,in_,ln,o);
		return o;
	}
	for (uint32_t i=0;i<ln-1;i++){
		_lstm_fwd(rnn->dt.cpu.lstm,in_+i*rnn->i);
	}
	float* o=_fc_fwd(rnn->dt.cpu.fc,_lstm_fwd(rnn->dt.cpu.lstm,in_+(ln-1)*rnn->i));
	_lstm_reset(rnn->dt.cpu.lstm);
	return o;
}



void lstm_rnn_train_multiple(LstmRnn rnn,Dataset dts,uint8_t e,uint32_t ln,uint32_t s){
	if (_bt==RNN_BACKEND_GPU){
		_gpu_lib_train_multiple(rnn,dts,e,ln,s);
	}
	else{
		for (uint8_t i=0;i<e;i++){
			uint8_t _lp=101;
			for (uint32_t j=0;j<ln;j++){
				if (_lp==101||((uint16_t)j)*100/ln>_lp){
					_lp=(uint8_t)(((uint16_t)j)*100/ln);
					printf("\x1b[0G\x1b[2KEpoch %hhu/%hhu: % 2hhu%%...",i+1,e,_lp);
				}
				lstm_rnn_train(rnn,dts+j,s,dts+j+1);
			}
			printf("\x1b[0G\x1b[2KEpoch %hhu/%hhu Complete\n",i+1,e);
		}
	}
}



void lstm_rnn_train(LstmRnn rnn,float* in_,uint32_t ln,float* t){
	if (_bt==RNN_BACKEND_GPU){
		_gpu_lib_train(rnn,in_,ln,t);
	}
	else{
		float** l=malloc(ln*sizeof(float*));
		for (uint32_t i=0;i<ln;i++){
			l[i]=_fc_train(rnn->dt.cpu.fc,_lstm_fwd_t(rnn->dt.cpu.lstm,in_+i*rnn->i),t+i*rnn->i,rnn->lr);
		}
		for (uint32_t i=ln;i>0;i--){
			_lstm_train(rnn->dt.cpu.lstm,l[i-1]);
			free(l[i-1]);
		}
		free(l);
		_lstm_update(rnn->dt.cpu.lstm,rnn->lr);
	}
}



void save_lstm_rnn(LstmRnn rnn){
	FILE* f=NULL;
	assert(fopen_s(&f,rnn->fp,"wb")==0);
	if (_bt==RNN_BACKEND_GPU){
		_gpu_lib_save(rnn,f);
	}
	else{
		assert(fwrite((void*)rnn->dt.cpu.lstm->bx,sizeof(float),rnn->dt.cpu.lstm->y,f)==rnn->dt.cpu.lstm->y);
		assert(fwrite((void*)rnn->dt.cpu.lstm->bf,sizeof(float),rnn->dt.cpu.lstm->y,f)==rnn->dt.cpu.lstm->y);
		assert(fwrite((void*)rnn->dt.cpu.lstm->bi,sizeof(float),rnn->dt.cpu.lstm->y,f)==rnn->dt.cpu.lstm->y);
		assert(fwrite((void*)rnn->dt.cpu.lstm->bo,sizeof(float),rnn->dt.cpu.lstm->y,f)==rnn->dt.cpu.lstm->y);
		assert(fwrite((void*)rnn->dt.cpu.lstm->wx,sizeof(float),rnn->dt.cpu.lstm->y*rnn->dt.cpu.lstm->_xy,f)==rnn->dt.cpu.lstm->y*rnn->dt.cpu.lstm->_xy);
		assert(fwrite((void*)rnn->dt.cpu.lstm->wf,sizeof(float),rnn->dt.cpu.lstm->y*rnn->dt.cpu.lstm->_xy,f)==rnn->dt.cpu.lstm->y*rnn->dt.cpu.lstm->_xy);
		assert(fwrite((void*)rnn->dt.cpu.lstm->wi,sizeof(float),rnn->dt.cpu.lstm->y*rnn->dt.cpu.lstm->_xy,f)==rnn->dt.cpu.lstm->y*rnn->dt.cpu.lstm->_xy);
		assert(fwrite((void*)rnn->dt.cpu.lstm->wo,sizeof(float),rnn->dt.cpu.lstm->y*rnn->dt.cpu.lstm->_xy,f)==rnn->dt.cpu.lstm->y*rnn->dt.cpu.lstm->_xy);
		assert(fwrite((void*)rnn->dt.cpu.fc->b,sizeof(float),rnn->dt.cpu.fc->y,f)==rnn->dt.cpu.fc->y);
		assert(fwrite((void*)rnn->dt.cpu.fc->w,sizeof(float),rnn->dt.cpu.fc->y*rnn->dt.cpu.fc->x,f)==rnn->dt.cpu.fc->y*rnn->dt.cpu.fc->x);
	}
	fclose(f);
}



void free_lstm_rnn(LstmRnn rnn){
	if (_bt==RNN_BACKEND_GPU){
		_gpu_lib_free(rnn);
	}
	else{
		free(rnn->dt.cpu.lstm->wx);
		free(rnn->dt.cpu.lstm->wf);
		free(rnn->dt.cpu.lstm->wi);
		free(rnn->dt.cpu.lstm->wo);
		free(rnn->dt.cpu.lstm->bx);
		free(rnn->dt.cpu.lstm->bf);
		free(rnn->dt.cpu.lstm->bi);
		free(rnn->dt.cpu.lstm->bo);
		if (rnn->dt.cpu.lstm->_sz!=-1){
			assert(rnn->dt.cpu.lstm->_sz==0);
			free(rnn->dt.cpu.lstm->_hg);
			free(rnn->dt.cpu.lstm->_cg);
			free(rnn->dt.cpu.lstm->_wxg);
			free(rnn->dt.cpu.lstm->_wfg);
			free(rnn->dt.cpu.lstm->_wig);
			free(rnn->dt.cpu.lstm->_wog);
			free(rnn->dt.cpu.lstm->_bxg);
			free(rnn->dt.cpu.lstm->_bfg);
			free(rnn->dt.cpu.lstm->_big);
			free(rnn->dt.cpu.lstm->_bog);
		}
		free(rnn->dt.cpu.lstm->_c);
		free(rnn->dt.cpu.lstm->_h);
		free(rnn->dt.cpu.lstm);
		free(rnn->dt.cpu.fc->w);
		free(rnn->dt.cpu.fc->b);
		free(rnn->dt.cpu.fc);
	}
	free(rnn);
}



void free_dataset(Dataset dts){
	if (_bt==RNN_BACKEND_GPU){
		_gpu_lib_free_dataset(dts);
	}
}
