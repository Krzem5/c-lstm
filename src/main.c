#include <lstm-rnn.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <windows.h>



#define HIDDEN_NODES 150
#define DATASET_ENTRIES 1000
#define SEQ_LEN 20
#define TOTAL_EPOCHS 10
#define DATASET_SIZE (size_t)(DATASET_ENTRIES-1+SEQ_LEN+1)
#define GRAPH_STRIDE 1
#define GRAPH_HEIGHT 20



static float DATA[DATASET_SIZE];
static Dataset dts=NULL;



void _graph_rnn(LstmRnn rnn){
	CONSOLE_SCREEN_BUFFER_INFO sbi;
	GetConsoleScreenBufferInfo(GetStdHandle(-11),&sbi);
	uint16_t w=sbi.dwMaximumWindowSize.X-6;
	assert(w>SEQ_LEN+1);
	size_t s=(size_t)(((float)rand())/RAND_MAX*(DATASET_SIZE-GRAPH_STRIDE*w-1));
	float mn=0;
	float mx=0;
	float* pl=malloc((w*GRAPH_STRIDE-SEQ_LEN)/GRAPH_STRIDE*sizeof(float));
	for (size_t i=0;i<w;i++){
		if (i==0||DATA[s+i*GRAPH_STRIDE]<mn){
			mn=DATA[s+i*GRAPH_STRIDE];
		}
		if (i==0||DATA[s+i*GRAPH_STRIDE]>mx){
			mx=DATA[s+i*GRAPH_STRIDE];
		}
		if (i*GRAPH_STRIDE>SEQ_LEN){
			float* p=lstm_rnn_predict_dataset(rnn,dts+s+i*GRAPH_STRIDE-SEQ_LEN,SEQ_LEN);
			*(pl+(i*GRAPH_STRIDE-SEQ_LEN)/GRAPH_STRIDE)=*p;
			if (*p<mn){
				mn=*p;
			}
			if (*p>mx){
				mx=*p;
			}
			free(p);
		}
	}
	char* tmp=malloc(1024*sizeof(char));
	size_t ln=sprintf_s(tmp,1024,"%llu%llu%f%f",s,s+w*GRAPH_STRIDE,mn,mx)+54;
	free(tmp);
	uint8_t* g=malloc(w*2*sizeof(uint8_t));
	for (size_t i=0;i<w;i++){
		g[i]=(uint8_t)(GRAPH_HEIGHT-roundf((DATA[s+i*GRAPH_STRIDE]-mn)/(mx-mn)*(GRAPH_HEIGHT-1))-1);
		if (i*GRAPH_STRIDE>SEQ_LEN){
			g[i+w]=(uint8_t)(GRAPH_HEIGHT-roundf((*(pl+(i*GRAPH_STRIDE-SEQ_LEN)/GRAPH_STRIDE)-mn)/(mx-mn)*(GRAPH_HEIGHT-1))-1);
		}
		else{
			g[i+w]=GRAPH_HEIGHT+1;
		}
	}
	printf("\x1b[1;1H\x1b[2J\x1b[48;2;24;24;24m");
	for (uint8_t i=0;i<w+6;i++){
		putchar(' ');
	}
	printf("\n\x1b[38;2;156;156;156m  ╔");
	for (uint8_t i=0;i<w;i++){
		printf("═");
	}
	printf("╗  \n  ║");
	for (uint8_t i=0;i<(w-ln)/2;i++){
		putchar(' ');
	}
	printf("\x1b[38;2;255;255;255mData Range\x1b[38;2;78;78;78m: \x1b[38;2;230;128;0m%llu \x1b[38;2;78;78;78m- \x1b[38;2;230;190;0m%llu\x1b[38;2;78;78;78m, \x1b[38;2;255;255;255mValue Range\x1b[38;2;78;78;78m: \x1b[38;2;160;50;230m%f \x1b[38;2;78;78;78m- \x1b[38;2;230;50;250m%f\x1b[38;2;78;78;78m, \x1b[38;2;255;255;255mGraph\x1b[38;2;78;78;78m: \x1b[38;2;50;100;210msin()\x1b[38;2;78;78;78m, \x1b[38;2;105;210;105mrnn()",s,s+w*GRAPH_STRIDE,mn,mx);
	for (uint8_t i=0;i<(w-ln+1)/2;i++){
		putchar(' ');
	}
	printf("\x1b[38;2;156;156;156m║  \n  ╠");
	for (uint8_t i=0;i<w;i++){
		printf("═");
	}
	printf("╣  \n");
	for (uint8_t i=0;i<GRAPH_HEIGHT;i++){
		if (g[0]!=i&&g[w]!=i){
			printf("\x1b[38;2;156;156;156m  ║");
		}
		else{
			printf("\x1b[38;2;156;156;156m  ╟");
		}
		for (uint8_t j=0;j<w;j++){
			if (j==SEQ_LEN/GRAPH_STRIDE+1&&g[j+w]==i){
				printf("\x1b[38;2;105;210;105m╶");
				continue;
			}
			if (j>SEQ_LEN/GRAPH_STRIDE+1){
				int8_t df=g[j+w]-g[j+w-1];
				if (g[j+w]==i){
					if (df>0){
						printf("\x1b[38;2;105;210;105m╰");
						continue;
					}
					else if (df<0){
						printf("\x1b[38;2;105;210;105m╭");
						continue;
					}
					else{
						printf("\x1b[38;2;105;210;105m─");
						continue;
					}
				}
				else if (g[j+w-1]==i){
					if (df>0){
						printf("\x1b[38;2;105;210;105m╮");
						continue;
					}
					else{
						printf("\x1b[38;2;105;210;105m╯");
						continue;
					}
				}
				else if ((df<0&&g[j+w]<i&&i<g[j+w-1])||(df>0&&g[j+w]>i&&i>g[j+w-1])){
					printf("\x1b[38;2;105;210;105m│");
					continue;
				}
			}
			if (j==0){
				if (g[j]==i){
					printf("\x1b[38;2;50;100;210m─");
					continue;
				}
			}
			else{
				int8_t df=g[j]-g[j-1];
				if (g[j]==i){
					if (df>0){
						printf("\x1b[38;2;50;100;210m╰");
					}
					else if (df<0){
						printf("\x1b[38;2;50;100;210m╭");
					}
					else{
						printf("\x1b[38;2;50;100;210m─");
					}
					continue;
				}
				else if (g[j-1]==i){
					if (df>0){
						printf("\x1b[38;2;50;100;210m╮");
						continue;
					}
					else{
						printf("\x1b[38;2;50;100;210m╯");
						continue;
					}
				}
				else if ((df<-1&&g[j]<i&&i<g[j-1])||(df>1&&g[j]>i&&i>g[j-1])){
					printf("\x1b[38;2;50;100;210m│");
					continue;
				}
			}
			putchar(' ');
		}
		if (g[w-1]!=i&&g[w*2-1]!=i){
			printf("\x1b[38;2;156;156;156m║  \n");
		}
		else{
			printf("\x1b[38;2;156;156;156m╢  \n");
		}
	}
	free(g);
	printf("  ╚");
	for (uint8_t i=0;i<w;i++){
		printf("═");
	}
	printf("╝  \n");
	for (uint8_t i=0;i<w+6;i++){
		putchar(' ');
	}
	printf("\x1b[0m\n");
}



int main(int argc,const char** argv){
	srand(1234/*(unsigned int)time(NULL)*/);
	for (size_t i=0;i<DATASET_SIZE;i++){
		DATA[i]=sinf(i*0.15f)*sinf(i*0.075f);
	}
	SetConsoleOutputCP(CP_UTF8);
	SetConsoleMode(GetStdHandle(-11),7);
	SetPriorityClass(GetCurrentProcess(),HIGH_PRIORITY_CLASS);
	if (set_rnn_backend(RNN_BACKEND_GPU)==false){
		return 1;
	}
	dts=create_dataset(DATA,DATASET_SIZE);
	LstmRnn rnn=init_lstm_rnn("../rnn-save3.rnn",1,HIDDEN_NODES,1,0.01f);
	// lstm_rnn_train_multiple(rnn,dts,TOTAL_EPOCHS,DATASET_ENTRIES,SEQ_LEN);
	// save_lstm_rnn(rnn);
	_graph_rnn(rnn);
	free_lstm_rnn(rnn);
	free_dataset(dts);
	return 0;
}
