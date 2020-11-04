#include <lstm-rnn.h>
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>



/****************************************/
const char* _s_sig="XYXYXYXY";
const char* _e_sig="ZWZWZWZW";
struct _MEM_BLOCK{
	struct _MEM_BLOCK* p;
	struct _MEM_BLOCK* n;
	void* ptr;
	size_t sz;
	unsigned int ln;
	const char* fn;
	bool f;
} _mem_head={
	NULL,
	NULL,
	NULL,
	0,
	0,
	NULL,
	false
};
void _dump_mem(void* s,size_t sz){
	printf("Memory Dump Of Address 0x%016llx - 0x%016llx (+ %llu):\n",(unsigned long long int)s,(unsigned long long int)s+sz,sz);
	size_t mx_n=8*(((sz+7)>>3)-1);
	unsigned char mx=1;
	while (mx_n>10){
		mx++;
		mx_n/=10;
	}
	char* f=malloc(mx+20);
	sprintf_s(f,mx+20,"0x%%016llx + %% %ullu: ",mx);
	for (size_t i=0;i<sz;i+=8){
		printf(f,(uintptr_t)s,(uintptr_t)i);
		unsigned char j;
		for (j=0;j<8;j++){
			if (i+j>=sz){
				break;
			}
			printf("%02x",*((unsigned char*)s+i+j));
			printf(" ");
		}
		if (j==0){
			break;
		}
		while (j<8){
			printf("   ");
			j++;
		}
		printf("| ");
		for (j=0;j<8;j++){
			if (i+j>=sz){
				break;
			}
			unsigned char c=*((unsigned char*)s+i+j);
			if (c>0x1f&&c!=0x7f){
				printf("%c  ",(char)c);
			}
			else{
				printf("%02x ",c);
			}
		}
		printf("\n");
	}
	free(f);
}
void _valid_mem(unsigned int ln,const char* fn){
	struct _MEM_BLOCK* n=&_mem_head;
	while (true){
		if (n->ptr!=NULL){
			for (unsigned char i=0;i<8;i++){
				if (*((char*)n->ptr+i)!=*(_s_sig+i)){
					printf("ERROR: Line %u (%s): Address 0x%016llx Allocated at Line %u (%s) has been Corrupted (0x%016llx-%u)!\n",ln,fn,((uint64_t)n->ptr+8),n->ln,n->fn,((uint64_t)n->ptr+8),8-i);
					_dump_mem(n->ptr,n->sz+16);
					raise(SIGABRT);
					return;
				}
			}
			for (unsigned char i=0;i<8;i++){
				if (*((char*)n->ptr+n->sz+i+8)!=*(_e_sig+i)){
					printf("ERROR: Line %u (%s): Address 0x%016llx Allocated at Line %u (%s) has been Corrupted (0x%016llx+%llu+%u)!\n",ln,fn,((uint64_t)n->ptr+8),n->ln,n->fn,((uint64_t)n->ptr+8),n->sz,i+1);
					_dump_mem(n->ptr,n->sz+16);
					raise(SIGABRT);
					return;
				}
			}
			if (n->f==true){
				bool ch=false;
				for (size_t i=0;i<n->sz;i++){
					if (*((unsigned char*)n->ptr+i+8)!=0xdd){
						if (ch==false){
							printf("ERROR: Line %u (%s): Detected Memory Change in Freed Block Allocated at Line %u (%s) (0x%016llx):",ln,fn,n->ln,n->fn,(uint64_t)n->ptr);
							ch=true;
						}
						else{
							printf(";");
						}
						printf(" +%llu (%02x)",i,*((unsigned char*)n->ptr+i+8));
					}
				}
				if (ch==true){
					printf("\n");
					_dump_mem(n->ptr,n->sz+16);
					raise(SIGABRT);
				}
			}
		}
		if (n->n==NULL){
			break;
		}
		n=n->n;
	}
}
void _get_mem_block(const void* ptr,unsigned int ln,const char* fn){
	_valid_mem(ln,fn);
	struct _MEM_BLOCK* n=&_mem_head;
	while ((uint64_t)ptr<(uint64_t)n->ptr||(uint64_t)ptr>(uint64_t)n->ptr+n->sz){
		if (n->n==NULL){
			printf("ERROR: Line %u (%s): Unknown Pointer 0x%016llx!\n",ln,fn,(uint64_t)ptr);
			raise(SIGABRT);
			return;
		}
		n=n->n;
	}
	printf("INFO:  Line %u (%s): Found Memory Block Containing 0x%016llx (+%llu) Allocated at Line %u (%s)!\n",ln,fn,(uint64_t)ptr,(uint64_t)ptr-(uint64_t)n->ptr,n->ln,n->fn);
	_dump_mem(n->ptr,n->sz+16);
}
bool _all_defined(const void* ptr,size_t e_sz,unsigned int ln,const char* fn){
	_valid_mem(ln,fn);
	struct _MEM_BLOCK* n=&_mem_head;
	while (n->ptr!=(unsigned char*)ptr-8){
		if (n->n==NULL){
			printf("ERROR: Line %u (%s): Unknown Pointer 0x%016llx!\n",ln,fn,(uint64_t)ptr);
			raise(SIGABRT);
			return false;
		}
		n=n->n;
	}
	assert((n->sz/e_sz)*e_sz==n->sz);
	bool e=false;
	for (size_t i=0;i<n->sz;i+=e_sz){
		bool f=true;
		for (size_t j=i;j<i+e_sz;j++){
			if (*((unsigned char*)ptr+j)!=0xcd){
				f=false;
				break;
			}
		}
		if (f==true){
			e=true;
			printf("ERROR: Line %u (%s): Found Uninitialised Memory Section in Pointer Allocated at Line %u (%s): 0x%016llx +%llu -> +%llu!\n",ln,fn,n->ln,n->fn,(uint64_t)ptr,i,i+e_sz);
		}
	}
	if (e==true){
		_dump_mem(n->ptr,n->sz+16);
		return false;
	}
	return true;
}
void* _malloc_mem(size_t sz,unsigned int ln,const char* fn){
	_valid_mem(ln,fn);
	if (sz<=0){
		printf("ERROR: Line %u (%s): Negative or Zero Size!\n",ln,fn);
		raise(SIGABRT);
		return NULL;
	}
	struct _MEM_BLOCK* n=&_mem_head;
	while (n->ptr!=NULL){
		if (n->n==NULL){
			n->n=malloc(sizeof(struct _MEM_BLOCK));
			n->n->p=NULL;
			n->n->n=NULL;
			n->n->ptr=NULL;
			n->n->sz=0;
			n->n->ln=0;
			n->n->fn=NULL;
			n->n->f=false;
		}
		n=n->n;
	}
	n->ptr=malloc(sz+16);
	if (n->ptr==NULL){
		printf("ERROR: Line %u (%s): Out of Memory!\n",ln,fn);
		raise(SIGABRT);
		return NULL;
	}
	for (size_t i=0;i<8;i++){
		*((char*)n->ptr+i)=*(_s_sig+i);
		*((char*)n->ptr+sz+i+8)=*(_e_sig+i);
	}
	n->sz=sz;
	n->ln=ln;
	n->fn=fn;
	n->f=false;
	return (void*)((uintptr_t)n->ptr+8);
}
void* _realloc_mem(const void* ptr,size_t sz,unsigned int ln,const char* fn){
	_valid_mem(ln,fn);
	if (ptr==NULL){
		return _malloc_mem(sz,ln,fn);
	}
	assert(sz>0);
	struct _MEM_BLOCK* n=&_mem_head;
	while (n->ptr!=(char*)ptr-8){
		if (n->n==NULL){
			printf("ERROR: Line %u (%s): Reallocating Unknown Pointer! (%p => %llu)\n",ln,fn,ptr,sz);
			raise(SIGABRT);
			break;
		}
		n=n->n;
	}
	if (n->f==true){
		printf("ERROR: Line %u (%s): Reallocating Freed Pointer! (%p => %llu)\n",ln,fn,ptr,sz);
		raise(SIGABRT);
		return NULL;
	}
	n->ptr=realloc(n->ptr,sz+16);
	if (n->ptr==NULL){
		printf("ERROR: Line %u (%s): Out of Memory! (%p => %llu)\n",ln,fn,ptr,sz);
		raise(SIGABRT);
		return NULL;
	}
	for (size_t i=0;i<8;i++){
		*((unsigned char*)n->ptr+i)=*(_s_sig+i);
		*((unsigned char*)n->ptr+sz+i+8)=*(_e_sig+i);
	}
	for (size_t i=n->sz;i<sz;i++){
		*((unsigned char*)n->ptr+i+8)=0xcd;
	}
	n->sz=sz;
	n->ln=ln;
	n->fn=fn;
	return (void*)((uintptr_t)n->ptr+8);
}
void _free_mem(const void* ptr,unsigned int ln,const char* fn){
	_valid_mem(ln,fn);
	struct _MEM_BLOCK* n=&_mem_head;
	while (n->ptr!=(char*)ptr-8){
		if (n->n==NULL){
			printf("ERROR: Line %u (%s): Freeing Unknown Pointer!\n",ln,fn);
			raise(SIGABRT);
			return;
		}
		n=n->n;
	}
	n->f=true;
	for (size_t i=0;i<n->sz;i++){
		*((unsigned char*)n->ptr+i+8)=0xdd;
	}
	// free(n->ptr);
	// n->ptr=NULL;
	// n->sz=0;
	// n->ln=0;
	// n->fn=NULL;
	// if (n->p!=NULL){
	// 	n->p->n=n->n;
	// 	if (n->n!=NULL){
	// 		n->n->p=n->p;
	// 	}
	// 	free(n);
	// }
}
#undef malloc
#define malloc(sz) _malloc_mem(sz,__LINE__,__func__)
#undef realloc
#define realloc(ptr,sz) _realloc_mem(ptr,sz,__LINE__,__func__)
#undef free
#define free(ptr) _free_mem(ptr,__LINE__,__func__)
#define check_mem() _valid_mem(__LINE__,__func__)
/****************************************/



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
	float* xh=malloc((lstm->x+lstm->y)*sizeof(float));
	for (uint8_t i=0;i<lstm->x;i++){
		xh[i]=in_[i];
	}
	for (uint8_t i=0;i<lstm->y;i++){
		xh[lstm->x+i]=lstm->_h[i];
	}
	for (uint8_t i=0;i<lstm->y;i++){
		float ca=lstm->bx[i];
		float f=lstm->bf[i];
		float i_=lstm->bi[i];
		float o=lstm->bo[i];
		for (uint16_t j=0;j<lstm->x+lstm->y;j++){
			ca+=lstm->wx[i][j]*xh[j];
			f+=lstm->wf[i][j]*xh[j];
			i_+=lstm->wi[i][j]*xh[j];
			o+=lstm->wo[i][j]*xh[j];
		}
		lstm->_c[i]=tanhf(ca)*sigmoidf(i_)+lstm->_c[i]*sigmoidf(f);
		lstm->_h[i]=tanhf(lstm->_c[i])*sigmoidf(o);
	}
	check_mem();
	return lstm->_h;
}



float* _lstm_fwd_t(struct __LSTMRNN_LSTM_LAYER* lstm,float* in_){
	if (lstm->_sz==-1){
		lstm->_sz=0;
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
		for (uint8_t i=0;i<lstm->y;i++){
			lstm->_hg[i]=0;
			lstm->_cg[i]=0;
			lstm->_wxg[i]=malloc((lstm->x+lstm->y)*sizeof(float));
			lstm->_wfg[i]=malloc((lstm->x+lstm->y)*sizeof(float));
			lstm->_wig[i]=malloc((lstm->x+lstm->y)*sizeof(float));
			lstm->_wog[i]=malloc((lstm->x+lstm->y)*sizeof(float));
			for (uint16_t j=0;j<lstm->x+lstm->y;j++){
				lstm->_wxg[i][j]=0;
				lstm->_wfg[i][j]=0;
				lstm->_wig[i][j]=0;
				lstm->_wog[i][j]=0;
			}
			lstm->_bxg[i]=0;
			lstm->_bfg[i]=0;
			lstm->_big[i]=0;
			lstm->_bog[i]=0;
		}
	}
	float* xh=malloc((lstm->x+lstm->y)*sizeof(float));
	for (uint8_t i=0;i<lstm->x;i++){
		xh[i]=in_[i];
	}
	for (uint8_t i=0;i<lstm->y;i++){
		xh[lstm->x+i]=lstm->_h[i];
	}
	lstm->_sz++;
	lstm->_cl=realloc(lstm->_cl,lstm->_sz*sizeof(float*));
	lstm->_xhl=realloc(lstm->_xhl,lstm->_sz*sizeof(float*));
	float* lc=malloc(lstm->y*sizeof(float));
	for (uint8_t i=0;i<lstm->y;i++){
		lc[i]=lstm->_c[i];
	}
	lstm->_cl[lstm->_sz-1]=lc;
	lstm->_xhl[lstm->_sz-1]=xh;
	float* ca=malloc(lstm->y*sizeof(float));
	float* f=malloc(lstm->y*sizeof(float));
	float* i_=malloc(lstm->y*sizeof(float));
	float* o=malloc(lstm->y*sizeof(float));
	float* out=malloc(lstm->y*sizeof(float));
	for (uint8_t i=0;i<lstm->y;i++){
		ca[i]=lstm->bx[i];
		f[i]=lstm->bf[i];
		i_[i]=lstm->bi[i];
		o[i]=lstm->bo[i];
		for (uint16_t j=0;j<lstm->x+lstm->y;j++){
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
	lstm->_cal=realloc(lstm->_cal,lstm->_sz*sizeof(float*));
	lstm->_fl=realloc(lstm->_fl,lstm->_sz*sizeof(float*));
	lstm->_il=realloc(lstm->_il,lstm->_sz*sizeof(float*));
	lstm->_ol=realloc(lstm->_ol,lstm->_sz*sizeof(float*));
	lstm->_outl=realloc(lstm->_outl,lstm->_sz*sizeof(float*));
	lstm->_cal[lstm->_sz-1]=ca;
	lstm->_fl[lstm->_sz-1]=f;
	lstm->_il[lstm->_sz-1]=i_;
	lstm->_ol[lstm->_sz-1]=o;
	lstm->_outl[lstm->_sz-1]=out;
	check_mem();
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
	float* hg=malloc(lstm->y*sizeof(float));
	for (uint8_t i=0;i<lstm->y;i++){
		hg[i]=0;
	}
	for (uint8_t i=0;i<lstm->y;i++){
		tg[i]+=lstm->_hg[i];
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
			if (j<lstm->x){
				/*og[j]+=lstm->wx[i][j]*lxg+lstm->wi[i][j]*lig+lstm->wf[i][j]*lfg+lstm->wo[i][j]*log;*/
			}
			else{
				hg[j-lstm->x]+=lstm->wx[i][j]*lxg+lstm->wi[i][j]*lig+lstm->wf[i][j]*lfg+lstm->wo[i][j]*log;
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
	free(lstm->_hg);
	lstm->_hg=hg;
	check_mem();
	// return og;
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
	check_mem();
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
	if (GetFileAttributesA(fp)==INVALID_FILE_ATTRIBUTES&&GetLastError()==ERROR_FILE_NOT_FOUND){
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
		assert(0);
	}
	return o;
}



float* lstm_rnn_predict(LstmRnn rnn,float** in_,uint32_t ln){
	for (uint32_t i=0;i<ln-1;i++){
		printf("AAA %lu\n",__LINE__);
		_lstm_fwd(rnn->lstm,in_[i]);
	}
	printf("AAA %lu\n",__LINE__);
	float* o=_fc_fwd(rnn->fc,_lstm_fwd(rnn->lstm,in_[ln-1]));
	printf("AAA %lu\n",__LINE__);
	_lstm_reset(rnn->lstm);
	printf("AAA %lu\n",__LINE__);
	check_mem();
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
	check_mem();
}



void save_lstm_rnn(LstmRnn rnn){
	// FILE* f=NULL;
	// assert(fopen_s(&f,rnn->fp,"wb")==0);
	// fclose(f);
}



void free_lstm_rnn(LstmRnn rnn){
	free(rnn->lstm);
	free(rnn->fc);
	free(rnn);
}
