#include <gpu_lstm_rnn.h>
#include <windows.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <ctime>



__device__ __forceinline__ float tanhf_k(float x){
	return (expf(2*x)-1)/(expf(2*x)+1);
}
__device__ __forceinline__ float sigmoidf_k(float x){
	return 1.0f/(1+expf(-x));
}
__device__ __forceinline__ float tanh_d_k(float x){
	return 1-x*x;
}
__device__ __forceinline__ float sigmoid_d_k(float x){
	return (1-x)*x;
}



__global__ void _init_rand(uint32_t s,curandState* rs){
	curand_init(s,blockIdx.x*blockDim.x+threadIdx.x,0,rs+blockIdx.x*blockDim.x+threadIdx.x);
}



__global__ void _fill_rand(curandState* rs,float* l,size_t ln,float a,float b){
	int s=blockIdx.x*blockDim.x+threadIdx.x;
	for (int i=s;i<ln;i+=blockDim.x*gridDim.x){
		l[i]=curand_uniform(rs+s)*(b-a)+a;
	}
}



__global__ void _clear_lstm_w_k(struct __LSTMRNN_LSTM_LAYER* lstm){
	for (uint16_t i=blockIdx.x*blockDim.x+threadIdx.x;i<lstm->y*lstm->_xy;i+=blockDim.x*gridDim.x){
		lstm->_wxg[i]=0;
		lstm->_wfg[i]=0;
		lstm->_wig[i]=0;
		lstm->_wog[i]=0;
	}
}



__global__ void _clear_lstm_b_k(struct __LSTMRNN_LSTM_LAYER* lstm){
	for (uint16_t i=blockIdx.x*blockDim.x+threadIdx.x;i<lstm->y;i+=blockDim.x*gridDim.x){
		lstm->_hg[i]=0;
		lstm->_cg[i]=0;
		lstm->_big[i]=0;
		lstm->_bog[i]=0;
		lstm->_bxg[i]=0;
		lstm->_bfg[i]=0;
		lstm->_big[i]=0;
		lstm->_bog[i]=0;
	}
}



__global__ void _clear_cfio(float* cfio,uint16_t l){
	for (uint16_t i=blockIdx.x*blockDim.x+threadIdx.x;i<l;i+=blockDim.x*gridDim.x){
		cfio[i]=0;
	}
}



__global__ void _clear_lstm_k(struct __LSTMRNN_LSTM_LAYER* lstm,uint32_t s){
	for (uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;i<lstm->y*s;i+=blockDim.x*gridDim.x){
		lstm->_cl[i]=0;
		lstm->_cal[i]=0;
		lstm->_fl[i]=0;
		lstm->_il[i]=0;
		lstm->_ol[i]=0;
		lstm->_outl[i]=0;
	}
}



__global__ void _clear_lstm2_k(struct __LSTMRNN_LSTM_LAYER* lstm,uint32_t s){
	for (uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;i<lstm->_xy*s;i+=blockDim.x*gridDim.x){
		lstm->_xhl[i]=0;
	}
}



__global__ void _clear_lstm_ch_k(struct __LSTMRNN_LSTM_LAYER* lstm){
	for (uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;i<lstm->y;i+=blockDim.x*gridDim.x){
		lstm->_c[i]=0;
		lstm->_h[i]=0;
	}
}



__global__ void _lstm_fwd_k(struct __LSTMRNN_LSTM_LAYER* lstm,float* in_,float* cfio){
	for (uint16_t i=blockIdx.x*blockDim.x+threadIdx.x;i<lstm->y*lstm->_xy;i+=blockDim.x*gridDim.x){
		uint16_t j=i%lstm->_xy;
		uint8_t k=i/lstm->_xy;
		float xh=(j<lstm->x?in_[j]:lstm->_h[j-lstm->x]);
		cfio[k*4]+=lstm->wx[i]*xh;
		cfio[k*4+1]+=lstm->wf[i]*xh;
		cfio[k*4+2]+=lstm->wi[i]*xh;
		cfio[k*4+3]+=lstm->wo[i]*xh;
	}
}



__global__ void _lstm_fwd_t_k(struct __LSTMRNN_LSTM_LAYER* lstm,float* in_,uint32_t k_){
	float* lc=lstm->_cl+k_*lstm->y;
	float* xh=lstm->_xhl+k_*lstm->_xy;
	float* ca=lstm->_cal+k_*lstm->y;
	float* f=lstm->_fl+k_*lstm->y;
	float* i_=lstm->_il+k_*lstm->y;
	float* o=lstm->_ol+k_*lstm->y;
	for (uint16_t i=blockIdx.x*blockDim.x+threadIdx.x;i<lstm->y*lstm->_xy;i+=blockDim.x*gridDim.x){
		uint16_t j=i%lstm->_xy;
		uint8_t k=i/lstm->_xy;
		if (k==0){
			if (j<lstm->x){
				xh[j]=in_[j];
			}
			else{
				xh[j]=lstm->_h[j-lstm->x];
				lc[j-lstm->x]=lstm->_c[j-lstm->x];
			}
		}
		ca[i]+=lstm->wx[i]*xh[j];
		f[i]+=lstm->wf[i]*xh[j];
		i_[i]+=lstm->wi[i]*xh[j];
		o[i]+=lstm->wo[i]*xh[j];
	}
}



__global__ void _lstm_fwd_sum_k(struct __LSTMRNN_LSTM_LAYER* lstm,float* cfio){
	for (uint8_t i=blockIdx.x*blockDim.x+threadIdx.x;i<lstm->y;i+=blockDim.x*gridDim.x){
		lstm->_c[i]=tanhf_k(cfio[i*4]+lstm->bx[i])*sigmoidf_k(cfio[i*4+2]+lstm->bi[i])+lstm->_c[i]*sigmoidf_k(cfio[i*4+1]+lstm->bf[i]);
		lstm->_h[i]=tanhf_k(lstm->_c[i])*sigmoidf_k(cfio[i*4+3]+lstm->bo[i]);
	}
}



__global__ void _lstm_fwd_sum_t_k(struct __LSTMRNN_LSTM_LAYER* lstm,uint32_t k){
	float* ca=lstm->_cal+k*lstm->y;
	float* f=lstm->_fl+k*lstm->y;
	float* i_=lstm->_il+k*lstm->y;
	float* o=lstm->_ol+k*lstm->y;
	float* out=lstm->_outl+k*lstm->y;
	for (uint8_t i=blockIdx.x*blockDim.x+threadIdx.x;i<lstm->y;i+=blockDim.x*gridDim.x){
		ca[i]=lstm->bx[i];
		f[i]=lstm->bf[i];
		i_[i]=lstm->bi[i];
		o[i]=lstm->bo[i];
		ca[i]=tanhf_k(ca[i]);
		f[i]=sigmoidf_k(f[i]);
		i_[i]=sigmoidf_k(i_[i]);
		o[i]=sigmoidf_k(o[i]);
		lstm->_c[i]=ca[i]*i_[i]+lstm->_c[i]*f[i];
		out[i]=tanhf_k(lstm->_c[i]);
		lstm->_h[i]=out[i]*o[i];
	}
}



__global__ void _fc_fwd_k(struct __LSTMRNN_FULLY_CONNECTED_LAYER* fc,float* in_,float* o){
	for (uint16_t i=blockIdx.x*blockDim.x+threadIdx.x;i<fc->y*fc->x;i+=blockDim.x*gridDim.x){
		o[i/fc->x]+=fc->w[i]*in_[i%fc->x];
	}
}



__global__ void _fc_fwd_sum_k(struct __LSTMRNN_FULLY_CONNECTED_LAYER* fc,float* o){
	for (uint8_t i=blockIdx.x*blockDim.x+threadIdx.x;i<fc->y;i+=blockDim.x*gridDim.x){
		o[i]+=fc->b[i];
	}
}



__global__ void _fc_train_sum_k(struct __LSTMRNN_FULLY_CONNECTED_LAYER* fc,float* p,float* tg,float lr){
	for (uint8_t i=blockIdx.x*blockDim.x+threadIdx.x;i<fc->y;i+=blockDim.x*gridDim.x){
		p[i]+=fc->b[i]-tg[i];
		fc->b[i]-=p[i]*lr;
	}
}



__global__ void _fc_train_k(struct __LSTMRNN_LSTM_LAYER* lstm,struct __LSTMRNN_FULLY_CONNECTED_LAYER* fc,float* in_,float* p,float lr,float* o){
	for (uint8_t i=blockIdx.x*blockDim.x+threadIdx.x;i<fc->y*fc->x;i+=blockDim.x*gridDim.x){
		uint8_t j=i%fc->x;
		uint8_t k=i/fc->x;
		if (j==0){
			o[k]=lstm->_hg[k];
			lstm->_hg[k]=0;
		}
		o[k]+=fc->w[i]*p[j];
		fc->w[i]-=p[j]*in_[j]*lr;
	}
}



__global__ void _lstm_train_k(struct __LSTMRNN_LSTM_LAYER* lstm,uint32_t k,float* tg){
	float* c=lstm->_cl+k*lstm->y;
	float* ca=lstm->_cal+k*lstm->y;
	float* f=lstm->_fl+k*lstm->y;
	float* i_=lstm->_il+k*lstm->y;
	float* o=lstm->_ol+k*lstm->y;
	float* out=lstm->_outl+k*lstm->y;
	for (uint8_t i=blockIdx.x*blockDim.x+threadIdx.x;i<lstm->y;i+=blockDim.x*gridDim.x){
		lstm->_cg[i]=tanh_d_k(out[i])*o[i]*tg[i]+lstm->_cg[i];
		c[i]=c[i]*lstm->_cg[i]*sigmoid_d_k(f[i]);// lfg = c[i]
		lstm->_cg[i]*=f[i];
		f[i]=tanh_d_k(ca[i])*i_[i]*lstm->_cg[i];// lxg = f[i]
		i_[i]=ca[i]*lstm->_cg[i]*sigmoid_d_k(i_[i]);// lig = i_[i]
		o[i]=out[i]*tg[i]*sigmoid_d_k(o[i]);// log = o[i]
		lstm->_bxg[i]+=f[i];
		lstm->_big[i]+=i_[i];
		lstm->_bfg[i]+=c[i];
		lstm->_bog[i]+=o[i];
	}
}



__global__ void _lstm_train_w_k(struct __LSTMRNN_LSTM_LAYER* lstm,uint32_t k){
	float* xh=lstm->_xhl+k*lstm->_xy;
	float* c=lstm->_cl+k*lstm->y;
	float* f=lstm->_fl+k*lstm->y;
	float* i_=lstm->_il+k*lstm->y;
	float* o=lstm->_ol+k*lstm->y;
	for (uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;i<lstm->y*lstm->_xy;i+=blockDim.x*gridDim.x){
		uint16_t j=i%lstm->_xy;
		uint8_t k=i/lstm->_xy;
		if (j>=lstm->x){
			lstm->_hg[j-lstm->x]+=lstm->wx[i]*f[k]+lstm->wi[i]*i_[k]+lstm->wf[i]*c[k]+lstm->wo[i]*o[k];
		}
		lstm->_wxg[i]+=f[k]*xh[j];
		lstm->_wig[i]+=i_[k]*xh[j];
		lstm->_wfg[i]+=c[k]*xh[j];
		lstm->_wog[i]+=o[k]*xh[j];
	}
}



__global__ void _lstm_update_w_k(struct __LSTMRNN_LSTM_LAYER* lstm,float lr){
	for (uint16_t i=blockIdx.x*blockDim.x+threadIdx.x;i<lstm->y*lstm->_xy;i+=blockDim.x*gridDim.x){
		lstm->wx[i]-=lstm->_wxg[i]*lr;
		lstm->wf[i]-=lstm->_wfg[i]*lr;
		lstm->wi[i]-=lstm->_wig[i]*lr;
		lstm->wo[i]-=lstm->_wog[i]*lr;
		lstm->_wxg[i]=0;
		lstm->_wfg[i]=0;
		lstm->_wig[i]=0;
		lstm->_wog[i]=0;
	}
}



__global__ void _lstm_update_b_k(struct __LSTMRNN_LSTM_LAYER* lstm,float lr){
	for (uint16_t i=blockIdx.x*blockDim.x+threadIdx.x;i<lstm->y;i+=blockDim.x*gridDim.x){
		lstm->_c[i]=0;
		lstm->_h[i]=0;
		lstm->_hg[i]=0;
		lstm->_cg[i]=0;
		lstm->bx[i]-=lstm->_bxg[i]*lr;
		lstm->bf[i]-=lstm->_bfg[i]*lr;
		lstm->bi[i]-=lstm->_big[i]*lr;
		lstm->bo[i]-=lstm->_bog[i]*lr;
		lstm->_big[i]=0;
		lstm->_bog[i]=0;
		lstm->_bxg[i]=0;
		lstm->_bfg[i]=0;
		lstm->_big[i]=0;
		lstm->_bog[i]=0;
	}
}



void gpu_lstm_rnn_setup_lib(void){
	cudaSetDevice(0);
}



Dataset gpu_lstm_rnn_create_dataset(float* dt,size_t sz){
	Dataset o=NULL;
	CUDA_CALL(cudaMalloc(&o,sz*sizeof(float)));
	CUDA_CALL(cudaMemcpy(o,dt,sz*sizeof(float),cudaMemcpyHostToDevice));
	return o;
}



void gpu_lstm_rnn_init_rand(LstmRnn rnn){
	rnn->dt.gpu.lstm=(struct __LSTMRNN_LSTM_LAYER*)malloc(sizeof(struct __LSTMRNN_LSTM_LAYER));
	rnn->dt.gpu.lstm->x=rnn->i;
	rnn->dt.gpu.lstm->y=rnn->h;
	rnn->dt.gpu.lstm->wx=NULL;
	rnn->dt.gpu.lstm->wf=NULL;
	rnn->dt.gpu.lstm->wi=NULL;
	rnn->dt.gpu.lstm->wo=NULL;
	rnn->dt.gpu.lstm->bx=NULL;
	rnn->dt.gpu.lstm->bf=NULL;
	rnn->dt.gpu.lstm->bi=NULL;
	rnn->dt.gpu.lstm->bo=NULL;
	rnn->dt.gpu.lstm->_xy=rnn->dt.gpu.lstm->x+rnn->dt.gpu.lstm->y;
	rnn->dt.gpu.lstm->_cl=NULL;
	rnn->dt.gpu.lstm->_xhl=NULL;
	rnn->dt.gpu.lstm->_cal=NULL;
	rnn->dt.gpu.lstm->_fl=NULL;
	rnn->dt.gpu.lstm->_il=NULL;
	rnn->dt.gpu.lstm->_ol=NULL;
	rnn->dt.gpu.lstm->_outl=NULL;
	rnn->dt.gpu.lstm->_c=NULL;
	rnn->dt.gpu.lstm->_h=NULL;
	rnn->dt.gpu.lstm->_hg=NULL;
	rnn->dt.gpu.lstm->_cg=NULL;
	rnn->dt.gpu.lstm->_wxg=NULL;
	rnn->dt.gpu.lstm->_wfg=NULL;
	rnn->dt.gpu.lstm->_wig=NULL;
	rnn->dt.gpu.lstm->_wog=NULL;
	rnn->dt.gpu.lstm->_bxg=NULL;
	rnn->dt.gpu.lstm->_bfg=NULL;
	rnn->dt.gpu.lstm->_big=NULL;
	rnn->dt.gpu.lstm->_bog=NULL;
	rnn->dt.gpu.fc=(struct __LSTMRNN_FULLY_CONNECTED_LAYER*)malloc(sizeof(struct __LSTMRNN_FULLY_CONNECTED_LAYER));
	rnn->dt.gpu.fc->x=rnn->h;
	rnn->dt.gpu.fc->y=rnn->o;
	rnn->dt.gpu.fc->w=NULL;
	rnn->dt.gpu.fc->b=NULL;
	rnn->dt.gpu.cfio=NULL;
	rnn->dt.gpu.to=NULL;
	CUDA_CALL(cudaMalloc(&rnn->dt.gpu.lstm->wx,rnn->dt.gpu.lstm->y*rnn->dt.gpu.lstm->_xy*sizeof(float)));
	CUDA_CALL(cudaMalloc(&rnn->dt.gpu.lstm->wf,rnn->dt.gpu.lstm->y*rnn->dt.gpu.lstm->_xy*sizeof(float)));
	CUDA_CALL(cudaMalloc(&rnn->dt.gpu.lstm->wi,rnn->dt.gpu.lstm->y*rnn->dt.gpu.lstm->_xy*sizeof(float)));
	CUDA_CALL(cudaMalloc(&rnn->dt.gpu.lstm->wo,rnn->dt.gpu.lstm->y*rnn->dt.gpu.lstm->_xy*sizeof(float)));
	CUDA_CALL(cudaMalloc(&rnn->dt.gpu.lstm->bx,rnn->dt.gpu.lstm->y*sizeof(float)));
	CUDA_CALL(cudaMalloc(&rnn->dt.gpu.lstm->bf,rnn->dt.gpu.lstm->y*sizeof(float)));
	CUDA_CALL(cudaMalloc(&rnn->dt.gpu.lstm->bi,rnn->dt.gpu.lstm->y*sizeof(float)));
	CUDA_CALL(cudaMalloc(&rnn->dt.gpu.lstm->bo,rnn->dt.gpu.lstm->y*sizeof(float)));
	CUDA_CALL(cudaMalloc(&rnn->dt.gpu.lstm->_c,rnn->dt.gpu.lstm->y*sizeof(float)));
	CUDA_CALL(cudaMalloc(&rnn->dt.gpu.lstm->_h,rnn->dt.gpu.lstm->y*sizeof(float)));
	CUDA_CALL(cudaMalloc(&rnn->dt.gpu.fc->w,rnn->dt.gpu.fc->y*rnn->dt.gpu.fc->x*sizeof(float)));
	CUDA_CALL(cudaMalloc(&rnn->dt.gpu.fc->b,rnn->dt.gpu.fc->y*sizeof(float)));
	CUDA_CALL(cudaMalloc(&rnn->dt.gpu.cfio,rnn->dt.gpu.lstm->y*4*sizeof(float)));
	CUDA_CALL(cudaMalloc(&rnn->dt.gpu.to,rnn->dt.gpu.fc->y*sizeof(float)));
	curandState* rs=NULL;
	uint16_t lstmw_n=(rnn->dt.gpu.lstm->y*rnn->dt.gpu.lstm->_xy+BLK_SIZE-1)/BLK_SIZE;
	uint16_t lstmb_n=(rnn->dt.gpu.lstm->y+BLK_SIZE-1)/BLK_SIZE;
	uint16_t fc_n=(rnn->dt.gpu.fc->y+BLK_SIZE-1)/BLK_SIZE;
	CUDA_CALL(cudaMalloc(&rs,(lstmw_n>fc_n?lstmw_n:fc_n)*BLK_SIZE*sizeof(curandState)));
	CUDA_GPU_CALL_CUSTOM(_init_rand,(lstmw_n>fc_n?lstmw_n:fc_n),BLK_SIZE,time(NULL),rs);
	CUDA_GPU_CALL_CUSTOM(_fill_rand,lstmw_n,BLK_SIZE,rs,rnn->dt.gpu.lstm->wx,rnn->dt.gpu.lstm->y*rnn->dt.gpu.lstm->_xy,0.1f,-0.1f);
	CUDA_GPU_CALL_CUSTOM(_fill_rand,lstmw_n,BLK_SIZE,rs,rnn->dt.gpu.lstm->wf,rnn->dt.gpu.lstm->y*rnn->dt.gpu.lstm->_xy,0.1f,-0.1f);
	CUDA_GPU_CALL_CUSTOM(_fill_rand,lstmw_n,BLK_SIZE,rs,rnn->dt.gpu.lstm->wi,rnn->dt.gpu.lstm->y*rnn->dt.gpu.lstm->_xy,0.1f,-0.1f);
	CUDA_GPU_CALL_CUSTOM(_fill_rand,lstmw_n,BLK_SIZE,rs,rnn->dt.gpu.lstm->wo,rnn->dt.gpu.lstm->y*rnn->dt.gpu.lstm->_xy,0.1f,-0.1f);
	CUDA_GPU_CALL_CUSTOM(_fill_rand,lstmb_n,BLK_SIZE,rs,rnn->dt.gpu.lstm->bx,rnn->dt.gpu.lstm->y,0.1f,-0.1f);
	CUDA_GPU_CALL_CUSTOM(_fill_rand,lstmb_n,BLK_SIZE,rs,rnn->dt.gpu.lstm->bf,rnn->dt.gpu.lstm->y,0.1f,-0.1f);
	CUDA_GPU_CALL_CUSTOM(_fill_rand,lstmb_n,BLK_SIZE,rs,rnn->dt.gpu.lstm->bi,rnn->dt.gpu.lstm->y,0.1f,-0.1f);
	CUDA_GPU_CALL_CUSTOM(_fill_rand,lstmb_n,BLK_SIZE,rs,rnn->dt.gpu.lstm->bo,rnn->dt.gpu.lstm->y,0.1f,-0.1f);
	CUDA_GPU_CALL_CUSTOM(_fill_rand,fc_n,BLK_SIZE,rs,rnn->dt.gpu.fc->w,rnn->dt.gpu.fc->y*rnn->dt.gpu.fc->x,1,-1);
	CUDA_CALL(cudaMemset(rnn->dt.gpu.lstm->_c,0,rnn->dt.gpu.lstm->y));
	CUDA_CALL(cudaMemset(rnn->dt.gpu.lstm->_c,0,rnn->dt.gpu.lstm->y));
	CUDA_CALL(cudaMemset(rnn->dt.gpu.fc->b,0,rnn->dt.gpu.fc->y));
	CUDA_CALL(cudaDeviceSynchronize());
	CUDA_CALL(cudaFree(rs));
	CUDA_CALL(cudaMalloc(&rnn->dt.gpu.lstm_d,sizeof(struct __LSTMRNN_LSTM_LAYER)));
	CUDA_CALL(cudaMemcpy(rnn->dt.gpu.lstm_d,rnn->dt.gpu.lstm,sizeof(struct __LSTMRNN_LSTM_LAYER),cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMalloc(&rnn->dt.gpu.fc_d,sizeof(struct __LSTMRNN_FULLY_CONNECTED_LAYER)));
	CUDA_CALL(cudaMemcpy(rnn->dt.gpu.fc_d,rnn->dt.gpu.fc,sizeof(struct __LSTMRNN_FULLY_CONNECTED_LAYER),cudaMemcpyHostToDevice));
}



void gpu_lstm_rnn_init_file(LstmRnn rnn,FILE* f){
	/**/
}



void gpu_lstm_rnn_predict_dataset(LstmRnn rnn,Dataset in_,uint32_t ln,float* o){
	for (uint32_t i=0;i<ln;i++){
		CUDA_GPU_CALL(_clear_cfio,rnn->dt.gpu.lstm->y*4,rnn->dt.gpu.cfio,rnn->dt.gpu.lstm->y*4);
		CUDA_CALL(cudaDeviceSynchronize());
		CUDA_GPU_CALL(_lstm_fwd_k,rnn->dt.gpu.lstm->y*rnn->dt.gpu.lstm->_xy,rnn->dt.gpu.lstm_d,in_+i*rnn->i,rnn->dt.gpu.cfio);
		CUDA_CALL(cudaDeviceSynchronize());
		CUDA_GPU_CALL(_lstm_fwd_sum_k,rnn->dt.gpu.lstm->y,rnn->dt.gpu.lstm_d,rnn->dt.gpu.cfio);
		CUDA_CALL(cudaDeviceSynchronize());
	}
	CUDA_GPU_CALL(_fc_fwd_k,rnn->dt.gpu.fc->y*rnn->dt.gpu.fc->x,rnn->dt.gpu.fc_d,rnn->dt.gpu.lstm->_h,rnn->dt.gpu.to);
	CUDA_CALL(cudaDeviceSynchronize());
	CUDA_GPU_CALL(_fc_fwd_sum_k,rnn->dt.gpu.fc->x,rnn->dt.gpu.fc_d,rnn->dt.gpu.to);
	CUDA_GPU_CALL(_clear_lstm_ch_k,rnn->dt.gpu.lstm->y,rnn->dt.gpu.lstm_d);
	CUDA_CALL(cudaDeviceSynchronize());
	CUDA_CALL(cudaMemcpy(o,rnn->dt.gpu.to,rnn->dt.gpu.fc->y*sizeof(float),cudaMemcpyDeviceToHost));
}



void gpu_lstm_rnn_predict(LstmRnn rnn,float* in_,uint32_t ln,float* o){
	float* tin=NULL;
	CUDA_CALL(cudaMalloc(&tin,ln*rnn->i*sizeof(float)));
	CUDA_CALL(cudaMemcpy(tin,in_,ln*rnn->i*sizeof(float),cudaMemcpyHostToDevice));
	for (uint32_t i=0;i<ln;i++){
		CUDA_GPU_CALL(_clear_cfio,rnn->dt.gpu.lstm->y*4,rnn->dt.gpu.cfio,rnn->dt.gpu.lstm->y*4);
		CUDA_CALL(cudaDeviceSynchronize());
		CUDA_GPU_CALL(_lstm_fwd_k,rnn->dt.gpu.lstm->y*rnn->dt.gpu.lstm->_xy,rnn->dt.gpu.lstm_d,tin+i*rnn->i,rnn->dt.gpu.cfio);
		CUDA_CALL(cudaDeviceSynchronize());
		CUDA_GPU_CALL(_lstm_fwd_sum_k,rnn->dt.gpu.lstm->y,rnn->dt.gpu.lstm_d,rnn->dt.gpu.cfio);
		CUDA_CALL(cudaDeviceSynchronize());
	}
	CUDA_CALL(cudaFree(tin));
	CUDA_GPU_CALL(_fc_fwd_k,rnn->dt.gpu.fc->x,rnn->dt.gpu.fc_d,rnn->dt.gpu.lstm->_h,rnn->dt.gpu.to);
	CUDA_GPU_CALL(_clear_lstm_ch_k,rnn->dt.gpu.lstm->y,rnn->dt.gpu.lstm_d);
	CUDA_CALL(cudaDeviceSynchronize());
	CUDA_CALL(cudaMemcpy(o,rnn->dt.gpu.to,rnn->dt.gpu.fc->y*sizeof(float),cudaMemcpyDeviceToHost));
}



void gpu_lstm_rnn_train_multiple(LstmRnn rnn,Dataset dts,uint8_t e,uint32_t ln,uint32_t s){
	if (rnn->dt.gpu.lstm->_hg==NULL){
		CUDA_CALL(cudaMalloc(&rnn->dt.gpu.lstm->_hg,rnn->dt.gpu.lstm->y*sizeof(float)));
		CUDA_CALL(cudaMalloc(&rnn->dt.gpu.lstm->_cg,rnn->dt.gpu.lstm->y*sizeof(float)));
		CUDA_CALL(cudaMalloc(&rnn->dt.gpu.lstm->_wxg,rnn->dt.gpu.lstm->y*rnn->dt.gpu.lstm->_xy*sizeof(float)));
		CUDA_CALL(cudaMalloc(&rnn->dt.gpu.lstm->_wfg,rnn->dt.gpu.lstm->y*rnn->dt.gpu.lstm->_xy*sizeof(float)));
		CUDA_CALL(cudaMalloc(&rnn->dt.gpu.lstm->_wig,rnn->dt.gpu.lstm->y*rnn->dt.gpu.lstm->_xy*sizeof(float)));
		CUDA_CALL(cudaMalloc(&rnn->dt.gpu.lstm->_wog,rnn->dt.gpu.lstm->y*rnn->dt.gpu.lstm->_xy*sizeof(float)));
		CUDA_CALL(cudaMalloc(&rnn->dt.gpu.lstm->_bxg,rnn->dt.gpu.lstm->y*sizeof(float)));
		CUDA_CALL(cudaMalloc(&rnn->dt.gpu.lstm->_bfg,rnn->dt.gpu.lstm->y*sizeof(float)));
		CUDA_CALL(cudaMalloc(&rnn->dt.gpu.lstm->_big,rnn->dt.gpu.lstm->y*sizeof(float)));
		CUDA_CALL(cudaMalloc(&rnn->dt.gpu.lstm->_bog,rnn->dt.gpu.lstm->y*sizeof(float)));
		CUDA_CALL(cudaMalloc(&rnn->dt.gpu.lstm->_cl,rnn->dt.gpu.lstm->y*s*sizeof(float)));
		CUDA_CALL(cudaMalloc(&rnn->dt.gpu.lstm->_xhl,rnn->dt.gpu.lstm->_xy*s*sizeof(float)));
		CUDA_CALL(cudaMalloc(&rnn->dt.gpu.lstm->_cal,rnn->dt.gpu.lstm->y*s*sizeof(float)));
		CUDA_CALL(cudaMalloc(&rnn->dt.gpu.lstm->_fl,rnn->dt.gpu.lstm->y*s*sizeof(float)));
		CUDA_CALL(cudaMalloc(&rnn->dt.gpu.lstm->_il,rnn->dt.gpu.lstm->y*s*sizeof(float)));
		CUDA_CALL(cudaMalloc(&rnn->dt.gpu.lstm->_ol,rnn->dt.gpu.lstm->y*s*sizeof(float)));
		CUDA_CALL(cudaMalloc(&rnn->dt.gpu.lstm->_outl,rnn->dt.gpu.lstm->y*s*sizeof(float)));
		CUDA_CALL(cudaMemcpy(rnn->dt.gpu.lstm_d,rnn->dt.gpu.lstm,sizeof(struct __LSTMRNN_LSTM_LAYER),cudaMemcpyHostToDevice));
		CUDA_GPU_CALL(_clear_lstm_w_k,rnn->dt.gpu.lstm->y*rnn->dt.gpu.lstm->_xy,rnn->dt.gpu.lstm_d);
		CUDA_GPU_CALL(_clear_lstm_b_k,rnn->dt.gpu.lstm->y,rnn->dt.gpu.lstm_d);
		CUDA_CALL(cudaDeviceSynchronize());
	}
	float* bl=NULL;
	float* fc_to=NULL;
	CUDA_CALL(cudaMalloc(&bl,rnn->dt.gpu.fc->x*s*sizeof(float)));
	CUDA_CALL(cudaMalloc(&fc_to,rnn->dt.gpu.fc->y*sizeof(float)));
	for (uint8_t i=0;i<e;i++){
		uint8_t _lp=0;
		for (uint32_t j=0;j<ln;j++){
			if (j==0||((uint16_t)j)*100/ln>_lp){
				_lp=(uint8_t)(((uint16_t)j)*100/ln);
				if (j==0){
					printf("Epoch %hhu/%hhu:  0%%...",i+1,e);
				}
				else{
					printf("\x1b[0G\x1b[2KEpoch %hhu/%hhu: % 2hhu%%...",i+1,e,_lp);
				}
			}
			CUDA_GPU_CALL(_clear_lstm_k,rnn->dt.gpu.lstm->y*s,rnn->dt.gpu.lstm_d,s);
			CUDA_GPU_CALL(_clear_lstm2_k,rnn->dt.gpu.lstm->_xy*s,rnn->dt.gpu.lstm_d,s);
			CUDA_CALL(cudaDeviceSynchronize());
			for (uint32_t k=0;k<s;k++){
				CUDA_GPU_CALL(_lstm_fwd_t_k,rnn->dt.gpu.lstm->y*rnn->dt.gpu.lstm->_xy,rnn->dt.gpu.lstm_d,dts+j*rnn->i+k*rnn->i,k);
				CUDA_CALL(cudaDeviceSynchronize());
				CUDA_GPU_CALL(_lstm_fwd_sum_t_k,rnn->dt.gpu.lstm->y,rnn->dt.gpu.lstm_d,k);
				CUDA_CALL(cudaDeviceSynchronize());
				CUDA_GPU_CALL(_fc_fwd_k,rnn->dt.gpu.fc->y*rnn->dt.gpu.fc->x,rnn->dt.gpu.fc_d,rnn->dt.gpu.lstm->_h,fc_to);
				CUDA_CALL(cudaDeviceSynchronize());
				CUDA_GPU_CALL(_fc_train_sum_k,rnn->dt.gpu.fc->x,rnn->dt.gpu.fc_d,fc_to,dts+j*rnn->i+k*rnn->i+rnn->i,rnn->lr);
				CUDA_CALL(cudaDeviceSynchronize());
				CUDA_GPU_CALL(_fc_train_k,rnn->dt.gpu.fc->y*rnn->dt.gpu.fc->x,rnn->dt.gpu.lstm_d,rnn->dt.gpu.fc_d,rnn->dt.gpu.lstm->_h,fc_to,rnn->lr,bl+k*rnn->dt.gpu.fc->x);
				CUDA_CALL(cudaDeviceSynchronize());
			}
			for (uint32_t k=s;k>0;k--){
				CUDA_GPU_CALL(_lstm_train_k,rnn->dt.gpu.lstm->y,rnn->dt.gpu.lstm_d,k-1,bl+(k-1)*rnn->dt.gpu.fc->x);
				CUDA_CALL(cudaDeviceSynchronize());
				CUDA_GPU_CALL(_lstm_train_w_k,rnn->dt.gpu.lstm->y*rnn->dt.gpu.lstm->_xy,rnn->dt.gpu.lstm_d,k-1);
				CUDA_CALL(cudaDeviceSynchronize());
			}
			CUDA_GPU_CALL(_lstm_update_w_k,rnn->dt.gpu.lstm->y*rnn->dt.gpu.lstm->_xy,rnn->dt.gpu.lstm_d,rnn->lr);
			CUDA_GPU_CALL(_lstm_update_b_k,rnn->dt.gpu.lstm->y,rnn->dt.gpu.lstm_d,rnn->lr);
			CUDA_CALL(cudaDeviceSynchronize());
		}
		printf("\x1b[0G\x1b[2KEpoch %hhu/%hhu Complete\n",i+1,e);
	}
	CUDA_CALL(cudaFree(bl));
	CUDA_CALL(cudaFree(fc_to));
}



void gpu_lstm_rnn_train(LstmRnn rnn,float* in_,uint32_t ln,float* t){
	/**/
}



void gpu_lstm_rnn_save(LstmRnn rnn,FILE* f){
	/**/
}



void gpu_lstm_rnn_free(LstmRnn rnn){
	CUDA_CALL(cudaFree(rnn->dt.gpu.lstm->wx));
	CUDA_CALL(cudaFree(rnn->dt.gpu.lstm->wf));
	CUDA_CALL(cudaFree(rnn->dt.gpu.lstm->wi));
	CUDA_CALL(cudaFree(rnn->dt.gpu.lstm->wo));
	CUDA_CALL(cudaFree(rnn->dt.gpu.lstm->bx));
	CUDA_CALL(cudaFree(rnn->dt.gpu.lstm->bf));
	CUDA_CALL(cudaFree(rnn->dt.gpu.lstm->bi));
	CUDA_CALL(cudaFree(rnn->dt.gpu.lstm->bo));
	CUDA_CALL(cudaFree(rnn->dt.gpu.lstm->_c));
	CUDA_CALL(cudaFree(rnn->dt.gpu.lstm->_h));
	if (rnn->dt.gpu.lstm->_hg!=NULL){
		CUDA_CALL(cudaFree(rnn->dt.gpu.lstm->_hg));
		CUDA_CALL(cudaFree(rnn->dt.gpu.lstm->_cg));
		CUDA_CALL(cudaFree(rnn->dt.gpu.lstm->_wxg));
		CUDA_CALL(cudaFree(rnn->dt.gpu.lstm->_wfg));
		CUDA_CALL(cudaFree(rnn->dt.gpu.lstm->_wig));
		CUDA_CALL(cudaFree(rnn->dt.gpu.lstm->_wog));
		CUDA_CALL(cudaFree(rnn->dt.gpu.lstm->_bxg));
		CUDA_CALL(cudaFree(rnn->dt.gpu.lstm->_bfg));
		CUDA_CALL(cudaFree(rnn->dt.gpu.lstm->_big));
		CUDA_CALL(cudaFree(rnn->dt.gpu.lstm->_bog));
		CUDA_CALL(cudaFree(rnn->dt.gpu.lstm->_cl));
		CUDA_CALL(cudaFree(rnn->dt.gpu.lstm->_xhl));
		CUDA_CALL(cudaFree(rnn->dt.gpu.lstm->_cal));
		CUDA_CALL(cudaFree(rnn->dt.gpu.lstm->_fl));
		CUDA_CALL(cudaFree(rnn->dt.gpu.lstm->_il));
		CUDA_CALL(cudaFree(rnn->dt.gpu.lstm->_ol));
		CUDA_CALL(cudaFree(rnn->dt.gpu.lstm->_outl));
	}
	CUDA_CALL(cudaFree(rnn->dt.gpu.fc->w));
	CUDA_CALL(cudaFree(rnn->dt.gpu.fc->b));
	CUDA_CALL(cudaFree(rnn->dt.gpu.lstm_d));
	CUDA_CALL(cudaFree(rnn->dt.gpu.fc_d));
	CUDA_CALL(cudaFree(rnn->dt.gpu.cfio));
	CUDA_CALL(cudaFree(rnn->dt.gpu.to));
	free(rnn->dt.gpu.lstm);
	free(rnn->dt.gpu.fc);
}



void gpu_lstm_rnn_free_dataset(Dataset dts){
	CUDA_CALL(cudaFree((float*)dts));
}



int WINAPI DllMain(void* dll,unsigned long r,void* rs){
	(void)dll;
	(void)r;
	(void)rs;
	return TRUE;
}
