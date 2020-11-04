#include <lstm-rnn.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <windows.h>



#define N_SEQ 200
#define SEQ_LEN 20
#define DATASET_SIZE (size_t)(N_SEQ-1+SEQ_LEN+1)
#define GRAPH_DATA_POINTS 100
#define GRAPH_HEIGHT 20



static float* DATA[DATASET_SIZE];



void _graph_rnn(LstmRnn rnn){
	assert(GRAPH_DATA_POINTS>SEQ_LEN+1);
	HANDLE ch=GetStdHandle(-11);
	DWORD cm=0;
	GetConsoleMode(ch,&cm);
	SetConsoleMode(ch,7);
	size_t s=(size_t)(((float)rand())/RAND_MAX*(DATASET_SIZE-GRAPH_DATA_POINTS-1));
	float mn=0;
	float mx=0;
	float* pl=malloc((GRAPH_DATA_POINTS-SEQ_LEN)*sizeof(float));
	for (size_t i=0;i<GRAPH_DATA_POINTS;i++){
		if (i==0||DATA[s+i][0]<mn){
			mn=DATA[s+i][0];
		}
		if (i==0||DATA[s+i][0]>mx){
			mx=DATA[s+i][0];
		}
		if (i>SEQ_LEN){
			float* p=lstm_rnn_predict(rnn,(float**)DATA+s+i-SEQ_LEN,SEQ_LEN);
			*(pl+i-SEQ_LEN)=*p;
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
	size_t ln=sprintf_s(tmp,1024,"%llu%llu%f%f",s,s+GRAPH_DATA_POINTS,mn,mx)+54;
	free(tmp);
	uint8_t g[GRAPH_DATA_POINTS][2];
	for (size_t i=0;i<GRAPH_DATA_POINTS;i++){
		g[i][0]=(uint8_t)(GRAPH_HEIGHT-roundf((DATA[s+i][0]-mn)/(mx-mn)*(GRAPH_HEIGHT-1))-1);
		if (i>SEQ_LEN){
			g[i][1]=(uint8_t)(GRAPH_HEIGHT-roundf((*(pl+i-SEQ_LEN)-mn)/(mx-mn)*(GRAPH_HEIGHT-1))-1);
		}
		else{
			g[i][1]=GRAPH_HEIGHT+1;
		}
	}
	printf("\x1b[48;2;24;24;24m");
	for (uint8_t i=0;i<GRAPH_DATA_POINTS+6;i++){
		putchar(' ');
	}
	printf("\n\x1b[38;2;156;156;156m  ┌");
	for (uint8_t i=0;i<GRAPH_DATA_POINTS;i++){
		printf("─");
	}
	printf("┐  \n  │");
	for (uint8_t i=0;i<(GRAPH_DATA_POINTS-ln)/2;i++){
		putchar(' ');
	}
	printf("\x1b[38;2;255;255;255mData Range\x1b[38;2;78;78;78m: \x1b[38;2;230;128;0m%llu \x1b[38;2;78;78;78m- \x1b[38;2;230;190;0m%llu\x1b[38;2;78;78;78m, \x1b[38;2;255;255;255mValue Range\x1b[38;2;78;78;78m: \x1b[38;2;160;50;230m%f \x1b[38;2;78;78;78m- \x1b[38;2;230;50;250m%f\x1b[38;2;78;78;78m, \x1b[38;2;255;255;255mGraph\x1b[38;2;78;78;78m: \x1b[38;2;50;100;210msin()\x1b[38;2;78;78;78m, \x1b[38;2;105;210;105mrnn()",s,s+GRAPH_DATA_POINTS,mn,mx);
	for (uint8_t i=0;i<(GRAPH_DATA_POINTS-ln+1)/2;i++){
		putchar(' ');
	}
	printf("\x1b[38;2;156;156;156m│  \n  ├");
	for (uint8_t i=0;i<GRAPH_DATA_POINTS;i++){
		printf("─");
	}
	printf("┤  \n");
	for (uint8_t i=0;i<GRAPH_HEIGHT;i++){
		printf("\x1b[38;2;156;156;156m  │");
		for (uint8_t j=0;j<GRAPH_DATA_POINTS;j++){
			if (g[j][1]==i){
				printf("\x1b[38;2;105;210;105m×");
			}
			else if (g[j][0]==i){
				printf("\x1b[38;2;50;100;210m·");
			}
			else{
				putchar(' ');
			}
		}
		printf("\x1b[38;2;156;156;156m│  \n");
	}
	printf("  └");
	for (uint8_t i=0;i<GRAPH_DATA_POINTS;i++){
		printf("─");
	}
	printf("┘  \n");
	for (uint8_t i=0;i<GRAPH_DATA_POINTS+6;i++){
		putchar(' ');
	}
	printf("\x1b[0m");
	SetConsoleMode(ch,cm);
}



int main(int argc,const char** argv){
	srand(0);
	for (size_t i=0;i<DATASET_SIZE;i++){
		DATA[i]=malloc(sizeof(float));
		DATA[i][0]=sinf(i*0.125f);
	}
	LstmRnn rnn=init_lstm_rnn("../rnn-save.rnn",1,150,1,0.01f);
	for (uint8_t i=0;i<1;i++){
		for (uint8_t j=0;j<1;j++){
			printf("%hu\n",i*1+j);
			lstm_rnn_train(rnn,DATA+j,SEQ_LEN,DATA+j+1);
		}
	}
	printf("BBB\n");
	_graph_rnn(rnn);
	save_lstm_rnn(rnn);
	free_lstm_rnn(rnn);
	for (size_t i=0;i<DATASET_SIZE;i++){
		free(DATA[i]);
	}
	return 0;
}
