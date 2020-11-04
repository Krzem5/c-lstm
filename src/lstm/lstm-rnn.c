#include <lstm-rnn.h>
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>



float tanh_d(float x){
	return 1-x*x;
}
float sigmoidf(float x){
	return 1.0f/(1+expf(-x));
}
float sigmoid_d(float x){
	return (1-x)*x;
}



float* _lstm_fwd(struct __LSTMRNN_LSTM_LAYER* lstm,float* in_){
	for (uint8_t i=0;i<lstm->y;i++){
		float ca=lstm->bx[i];
		float f=lstm->bf[i];
		float i_=lstm->bi[i];
		float o=lstm->bo[i];
		for (uint16_t j=0;j<lstm->x+lstm->y;j++){
			float xh=(j<lstm->x?in_[j]:lstm->_h[j-lstm->x]);
			ca+=lstm->wx[i][j]*xh;
			f+=lstm->wf[i][j]*xh;
			i_+=lstm->wi[i][j]*xh;
			o+=lstm->wo[i][j]*xh;
		}
		lstm->_c[i]=tanhf(ca)*sigmoidf(i_)+lstm->_c[i]*sigmoidf(f);
		lstm->_h[i]=tanhf(lstm->_c[i])*sigmoidf(o);
	}
	return lstm->_h;
}



float* _lstm_fwd_t(struct __LSTMRNN_LSTM_LAYER* lstm,float* in_){
	if (lstm->_sz==-1){
		lstm->_hg=malloc(lstm->y*sizeof(float));
		lstm->_cg=malloc(lstm->y*sizeof(float));
		lstm->_wxg=malloc(lstm->y*sizeof(float*));
		lstm->_wfg=malloc(lstm->y*sizeof(float*));
		lstm->_wig=malloc(lstm->y*sizeof(float*));
		lstm->_wog=malloc(lstm->y*sizeof(float*));
		lstm->_bxg=malloc(lstm->y*sizeof(float));
		lstm->_bfg=malloc(lstm->y*sizeof(float));
		lstm->_big=malloc(lstm->y*sizeof(float));
		lstm->_bog=malloc(lstm->y*sizeof(float));
	}
	float* lc=malloc(lstm->y*sizeof(float));
	float* xh=malloc((lstm->x+lstm->y)*sizeof(float));
	float* ca=malloc(lstm->y*sizeof(float));
	float* f=malloc(lstm->y*sizeof(float));
	float* i_=malloc(lstm->y*sizeof(float));
	float* o=malloc(lstm->y*sizeof(float));
	float* out=malloc(lstm->y*sizeof(float));
	for (uint8_t i=0;i<lstm->y;i++){
		if (lstm->_sz==-1){
			lstm->_hg[i]=0;
			lstm->_cg[i]=0;
			lstm->_wxg[i]=malloc((lstm->x+lstm->y)*sizeof(float));
			lstm->_wfg[i]=malloc((lstm->x+lstm->y)*sizeof(float));
			lstm->_wig[i]=malloc((lstm->x+lstm->y)*sizeof(float));
			lstm->_wog[i]=malloc((lstm->x+lstm->y)*sizeof(float));
			lstm->_bxg[i]=0;
			lstm->_bfg[i]=0;
			lstm->_big[i]=0;
			lstm->_bog[i]=0;
		}
		ca[i]=lstm->bx[i];
		f[i]=lstm->bf[i];
		i_[i]=lstm->bi[i];
		o[i]=lstm->bo[i];
		for (uint16_t j=0;j<lstm->x+lstm->y;j++){
			if (lstm->_sz==-1){
				if (i==lstm->y-1&&j==lstm->x+lstm->y-1){
					lstm->_sz=0;
				}
				lstm->_wxg[i][j]=0;
				lstm->_wfg[i][j]=0;
				lstm->_wig[i][j]=0;
				lstm->_wog[i][j]=0;
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
			ca[i]+=lstm->wx[i][j]*xh[j];
			f[i]+=lstm->wf[i][j]*xh[j];
			i_[i]+=lstm->wi[i][j]*xh[j];
			o[i]+=lstm->wo[i][j]*xh[j];
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



void _lstm_train(struct __LSTMRNN_LSTM_LAYER* lstm,float* tg){
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
		for (uint8_t j=0;j<lstm->x+lstm->y;j++){
			if (j>=lstm->x){
				if (i==0){
					if (j>lstm->x){
						tg[j-lstm->x]+=lstm->_hg[j-lstm->x];
					}
					lstm->_hg[j-lstm->x]=0;
				}
				lstm->_hg[j-lstm->x]+=lstm->wx[i][j]*lxg+lstm->wi[i][j]*lig+lstm->wf[i][j]*lfg+lstm->wo[i][j]*log;
			}
			lstm->_wxg[i][j]+=lxg*xh[j];
			lstm->_wig[i][j]+=lig*xh[j];
			lstm->_wfg[i][j]+=lfg*xh[j];
			lstm->_wog[i][j]+=log*xh[j];
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



void _lstm_update(struct __LSTMRNN_LSTM_LAYER* lstm,float lr){
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
		for (uint16_t j=0;j<lstm->x+lstm->y;j++){
			lstm->wx[i][j]-=lstm->_wxg[i][j]*lr;
			lstm->wf[i][j]-=lstm->_wfg[i][j]*lr;
			lstm->wi[i][j]-=lstm->_wig[i][j]*lr;
			lstm->wo[i][j]-=lstm->_wog[i][j]*lr;
			lstm->_wxg[i][j]=0;
			lstm->_wfg[i][j]=0;
			lstm->_wig[i][j]=0;
			lstm->_wog[i][j]=0;
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



void _lstm_reset(struct __LSTMRNN_LSTM_LAYER* lstm){
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
			o[i]+=fc->w[i][j]*in_[j];
		}
	}
	return o;
}



float* _fc_train(struct __LSTMRNN_FULLY_CONNECTED_LAYER* fc,float* in_,float* tg,float lr){
	float* o=malloc(fc->x*sizeof(float));
	for (uint8_t i=0;i<fc->y;i++){
		float p=fc->b[i];
		for (uint8_t j=0;j<fc->x;j++){
			p+=fc->w[i][j]*in_[j];
		}
		float bg=p-tg[i];
		fc->b[i]-=bg*lr;
		for (uint8_t j=0;j<fc->x;j++){
			if (i==0){
				o[j]=0;
			}
			o[j]+=fc->w[i][j]*bg;
			fc->w[i][j]-=bg*in_[j]*lr;
		}
	}
	return o;
}



float randf(void){
	return ((float)rand())/RAND_MAX;
}



LstmRnn init_lstm_rnn(const char* fp,uint8_t in,uint8_t hn,uint8_t on,float lr){
	LstmRnn o=malloc(sizeof(struct __LSTMRNN));
	o->fp=fp;
	o->i=in;
	o->h=hn;
	o->o=on;
	o->lr=lr;
	o->lstm=malloc(sizeof(struct __LSTMRNN_LSTM_LAYER));
	o->lstm->x=in;
	o->lstm->y=hn;
	o->lstm->wx=malloc(o->lstm->y*sizeof(float*));
	o->lstm->wf=malloc(o->lstm->y*sizeof(float*));
	o->lstm->wi=malloc(o->lstm->y*sizeof(float*));
	o->lstm->wo=malloc(o->lstm->y*sizeof(float*));
	o->lstm->bx=malloc(o->lstm->y*sizeof(float));
	o->lstm->bf=malloc(o->lstm->y*sizeof(float));
	o->lstm->bi=malloc(o->lstm->y*sizeof(float));
	o->lstm->bo=malloc(o->lstm->y*sizeof(float));
	o->lstm->_sz=-1;
	o->lstm->_cl=NULL;
	o->lstm->_xhl=NULL;
	o->lstm->_cal=NULL;
	o->lstm->_fl=NULL;
	o->lstm->_il=NULL;
	o->lstm->_ol=NULL;
	o->lstm->_outl=NULL;
	o->lstm->_c=malloc(o->lstm->y*sizeof(float));
	o->lstm->_h=malloc(o->lstm->y*sizeof(float));
	o->lstm->_hg=NULL;
	o->lstm->_cg=NULL;
	o->lstm->_wxg=NULL;
	o->lstm->_wfg=NULL;
	o->lstm->_wig=NULL;
	o->lstm->_wog=NULL;
	o->lstm->_bxg=NULL;
	o->lstm->_bfg=NULL;
	o->lstm->_big=NULL;
	o->lstm->_bog=NULL;
	o->fc=malloc(sizeof(struct __LSTMRNN_FULLY_CONNECTED_LAYER));
	o->fc->x=hn;
	o->fc->y=on;
	o->fc->w=malloc(o->fc->y*sizeof(float*));
	o->fc->b=malloc(o->fc->y*sizeof(float));
	if (GetFileAttributesA(o->fp)==INVALID_FILE_ATTRIBUTES&&GetLastError()==ERROR_FILE_NOT_FOUND){
		for (uint8_t i=0;i<o->lstm->y;i++){
			o->lstm->_c[i]=0;
			o->lstm->_h[i]=0;
			o->lstm->wx[i]=malloc((o->lstm->x+o->lstm->y)*sizeof(float));
			o->lstm->wf[i]=malloc((o->lstm->x+o->lstm->y)*sizeof(float));
			o->lstm->wi[i]=malloc((o->lstm->x+o->lstm->y)*sizeof(float));
			o->lstm->wo[i]=malloc((o->lstm->x+o->lstm->y)*sizeof(float));
			for (uint16_t j=0;j<o->lstm->x+o->lstm->y;j++){
				o->lstm->wx[i][j]=randf()*0.2f-0.1f;
				o->lstm->wf[i][j]=randf()*0.2f-0.1f;
				o->lstm->wi[i][j]=randf()*0.2f-0.1f;
				o->lstm->wo[i][j]=randf()*0.2f-0.1f;
			}
			o->lstm->bx[i]=randf()*0.2f-0.1f;
			o->lstm->bf[i]=randf()*0.2f-0.1f;
			o->lstm->bi[i]=randf()*0.2f-0.1f;
			o->lstm->bo[i]=randf()*0.2f-0.1f;
		}
		for (uint8_t i=0;i<o->fc->y;i++){
			o->fc->w[i]=malloc(o->fc->x*sizeof(float));
			for (uint8_t j=0;j<o->fc->x;j++){
				o->fc->w[i][j]=randf()*2-1;
			}
			o->fc->b[i]=0;
		}
	}
	else{
		FILE* f=NULL;
		assert(fopen_s(&f,o->fp,"rb")==0);
		assert(fread((void*)o->lstm->bx,sizeof(float),o->lstm->y,f)==o->lstm->y);
		assert(fread((void*)o->lstm->bf,sizeof(float),o->lstm->y,f)==o->lstm->y);
		assert(fread((void*)o->lstm->bi,sizeof(float),o->lstm->y,f)==o->lstm->y);
		assert(fread((void*)o->lstm->bo,sizeof(float),o->lstm->y,f)==o->lstm->y);
		for (uint8_t i=0;i<o->lstm->y;i++){
			o->lstm->_c[i]=0;
			o->lstm->_h[i]=0;
			o->lstm->wx[i]=malloc((o->lstm->x+o->lstm->y)*sizeof(float));
			o->lstm->wf[i]=malloc((o->lstm->x+o->lstm->y)*sizeof(float));
			o->lstm->wi[i]=malloc((o->lstm->x+o->lstm->y)*sizeof(float));
			o->lstm->wo[i]=malloc((o->lstm->x+o->lstm->y)*sizeof(float));
			assert(fread((void*)o->lstm->wx[i],sizeof(float),o->lstm->x+o->lstm->y,f)==o->lstm->x+o->lstm->y);
			assert(fread((void*)o->lstm->wf[i],sizeof(float),o->lstm->x+o->lstm->y,f)==o->lstm->x+o->lstm->y);
			assert(fread((void*)o->lstm->wi[i],sizeof(float),o->lstm->x+o->lstm->y,f)==o->lstm->x+o->lstm->y);
			assert(fread((void*)o->lstm->wo[i],sizeof(float),o->lstm->x+o->lstm->y,f)==o->lstm->x+o->lstm->y);
		}
		assert(fread((void*)o->fc->b,sizeof(float),o->fc->y,f)==o->fc->y);
		for (uint8_t i=0;i<o->fc->y;i++){
			o->fc->w[i]=malloc(o->fc->x*sizeof(float));
			assert(fread((void*)o->fc->w[i],sizeof(float),o->fc->x,f)==o->fc->x);
		}
		fclose(f);
	}
	return o;
}



float* lstm_rnn_predict(LstmRnn rnn,float** in_,uint32_t ln){
	for (uint32_t i=0;i<ln-1;i++){
		_lstm_fwd(rnn->lstm,in_[i]);
	}
	float* o=_fc_fwd(rnn->fc,_lstm_fwd(rnn->lstm,in_[ln-1]));
	_lstm_reset(rnn->lstm);
	return o;
}



void lstm_rnn_train(LstmRnn rnn,float** in_,uint32_t ln,float** t){
	float** l=malloc(ln*sizeof(float*));
	for (uint32_t i=0;i<ln;i++){
		l[i]=_fc_train(rnn->fc,_lstm_fwd_t(rnn->lstm,in_[i]),t[i],rnn->lr);
	}
	for (uint32_t i=ln;i>0;i--){
		_lstm_train(rnn->lstm,l[i-1]);
		free(l[i-1]);
	}
	free(l);
	_lstm_update(rnn->lstm,rnn->lr);
}



void save_lstm_rnn(LstmRnn rnn){
	FILE* f=NULL;
	assert(fopen_s(&f,rnn->fp,"wb")==0);
	assert(fwrite((void*)rnn->lstm->bx,sizeof(float),rnn->lstm->y,f)==rnn->lstm->y);
	assert(fwrite((void*)rnn->lstm->bf,sizeof(float),rnn->lstm->y,f)==rnn->lstm->y);
	assert(fwrite((void*)rnn->lstm->bi,sizeof(float),rnn->lstm->y,f)==rnn->lstm->y);
	assert(fwrite((void*)rnn->lstm->bo,sizeof(float),rnn->lstm->y,f)==rnn->lstm->y);
	for (uint8_t i=0;i<rnn->lstm->y;i++){
		assert(fwrite((void*)rnn->lstm->wx[i],sizeof(float),rnn->lstm->x+rnn->lstm->y,f)==rnn->lstm->x+rnn->lstm->y);
		assert(fwrite((void*)rnn->lstm->wf[i],sizeof(float),rnn->lstm->x+rnn->lstm->y,f)==rnn->lstm->x+rnn->lstm->y);
		assert(fwrite((void*)rnn->lstm->wi[i],sizeof(float),rnn->lstm->x+rnn->lstm->y,f)==rnn->lstm->x+rnn->lstm->y);
		assert(fwrite((void*)rnn->lstm->wo[i],sizeof(float),rnn->lstm->x+rnn->lstm->y,f)==rnn->lstm->x+rnn->lstm->y);
	}
	assert(fwrite((void*)rnn->fc->b,sizeof(float),rnn->fc->y,f)==rnn->fc->y);
	for (uint8_t i=0;i<rnn->fc->y;i++){
		assert(fwrite((void*)rnn->fc->w[i],sizeof(float),rnn->fc->x,f)==rnn->fc->x);
	}
	fclose(f);
}



void free_lstm_rnn(LstmRnn rnn){
	free(rnn->lstm);
	free(rnn->fc);
	free(rnn);
}
