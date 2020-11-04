#include <lstm-rnn.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <windows.h>



#define HIDDEN_NODES 150
#define DATASET_ENTRIES 1000
#define SEQ_LEN 20
#define TOTAL_EPOCHS 5
#define DATASET_SIZE (size_t)(DATASET_ENTRIES-1+SEQ_LEN+1)
#define GRAPH_DATA_POINTS 114
#define GRAPH_HEIGHT 20



static float* DATA[DATASET_SIZE];



void _graph_rnn(LstmRnn rnn){
	assert(GRAPH_DATA_POINTS>SEQ_LEN+1);
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
	printf("\x1b[1;1H\x1b[2J\x1b[48;2;24;24;24m");
	for (uint8_t i=0;i<GRAPH_DATA_POINTS+6;i++){
		putchar(' ');
	}
	printf("\n\x1b[38;2;156;156;156m  ╔");
	for (uint8_t i=0;i<GRAPH_DATA_POINTS;i++){
		printf("═");
	}
	printf("╗  \n  ║");
	for (uint8_t i=0;i<(GRAPH_DATA_POINTS-ln)/2;i++){
		putchar(' ');
	}
	printf("\x1b[38;2;255;255;255mData Range\x1b[38;2;78;78;78m: \x1b[38;2;230;128;0m%llu \x1b[38;2;78;78;78m- \x1b[38;2;230;190;0m%llu\x1b[38;2;78;78;78m, \x1b[38;2;255;255;255mValue Range\x1b[38;2;78;78;78m: \x1b[38;2;160;50;230m%f \x1b[38;2;78;78;78m- \x1b[38;2;230;50;250m%f\x1b[38;2;78;78;78m, \x1b[38;2;255;255;255mGraph\x1b[38;2;78;78;78m: \x1b[38;2;50;100;210msin()\x1b[38;2;78;78;78m, \x1b[38;2;105;210;105mrnn()",s,s+GRAPH_DATA_POINTS,mn,mx);
	for (uint8_t i=0;i<(GRAPH_DATA_POINTS-ln+1)/2;i++){
		putchar(' ');
	}
	printf("\x1b[38;2;156;156;156m║  \n  ╠");
	for (uint8_t i=0;i<GRAPH_DATA_POINTS;i++){
		printf("═");
	}
	printf("╣  \n");
	for (uint8_t i=0;i<GRAPH_HEIGHT;i++){
		if (g[0][0]!=i&&g[0][1]!=i){
			printf("\x1b[38;2;156;156;156m  ║");
		}
		else{
			printf("\x1b[38;2;156;156;156m  ╟");
		}
		for (uint8_t j=0;j<GRAPH_DATA_POINTS;j++){
			if (j==SEQ_LEN+1&&g[j][1]==i){
				printf("\x1b[38;2;105;210;105m╶");
				continue;
			}
			if (j>SEQ_LEN+1){
				int8_t df=g[j][1]-g[j-1][1];
				if (g[j][1]==i){
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
				else if (g[j-1][1]==i){
					if (df>0){
						printf("\x1b[38;2;105;210;105m╮");
						continue;
					}
					else{
						printf("\x1b[38;2;105;210;105m╯");
						continue;
					}
				}
				else if ((df<0&&g[j][1]<i&&i<g[j-1][1])||(df>0&&g[j][1]>i&&i>g[j-1][1])){
					printf("\x1b[38;2;105;210;105m│");
					continue;
				}
			}
			if (j==0){
				if (g[j][0]==i){
					printf("\x1b[38;2;50;100;210m─");
					continue;
				}
			}
			else{
				int8_t df=g[j][0]-g[j-1][0];
				if (g[j][0]==i){
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
				else if (g[j-1][0]==i){
					if (df>0){
						printf("\x1b[38;2;50;100;210m╮");
						continue;
					}
					else{
						printf("\x1b[38;2;50;100;210m╯");
						continue;
					}
				}
				else if ((df<-1&&g[j][0]<i&&i<g[j-1][0])||(df>1&&g[j][0]>i&&i>g[j-1][0])){
					printf("\x1b[38;2;50;100;210m│");
					continue;
				}
			}
			// printf("\x1b[38;2;50;100;210m·");
			// continue;
			putchar(' ');
		}
		if (g[GRAPH_DATA_POINTS-1][0]!=i&&g[GRAPH_DATA_POINTS-1][1]!=i){
			printf("\x1b[38;2;156;156;156m║  \n");
		}
		else{
			printf("\x1b[38;2;156;156;156m╢  \n");
		}
	}
	printf("  ╚");
	for (uint8_t i=0;i<GRAPH_DATA_POINTS;i++){
		printf("═");
	}
	printf("╝  \n");
	for (uint8_t i=0;i<GRAPH_DATA_POINTS+6;i++){
		putchar(' ');
	}
	printf("\x1b[0m");
}



int main(int argc,const char** argv){
	srand((unsigned int)time(0));
	for (size_t i=0;i<DATASET_SIZE;i++){
		DATA[i]=malloc(sizeof(float));
		DATA[i][0]=sinf(i*0.15f)*sinf(i*0.075f);
	}
	SetConsoleOutputCP(CP_UTF8);
	SetConsoleMode(GetStdHandle(-11),7);
	SetPriorityClass(GetCurrentProcess(),HIGH_PRIORITY_CLASS);
	if (set_rnn_backend(RNN_BACKEND_GPU)==false){
		return 1;
	}
	LstmRnn rnn=init_lstm_rnn("../rnn-save3.rnn",1,HIDDEN_NODES,1,0.01f);
	// for (uint8_t i=0;i<TOTAL_EPOCHS;i++){
	// 	uint8_t _lp=101;
	// 	for (uint32_t j=0;j<DATASET_ENTRIES;j++){
	// 		if (_lp==101||((uint16_t)j)*100/DATASET_ENTRIES>_lp){
	// 			_lp=(uint8_t)(((uint16_t)j)*100/DATASET_ENTRIES);
	// 			printf("\x1b[0G\x1b[2KEpoch %hhu/%hhu: % 2hhu%%...",i+1,TOTAL_EPOCHS,_lp);
	// 		}
	// 		lstm_rnn_train(rnn,DATA+j,SEQ_LEN,DATA+j+1);
	// 	}
	// 	printf("\x1b[0G\x1b[2KEpoch %hhu/%hhu Complete\n",i+1,TOTAL_EPOCHS);
	// }
	// save_lstm_rnn(rnn);
	// _graph_rnn(rnn);
	free_lstm_rnn(rnn);
	for (size_t i=0;i<DATASET_SIZE;i++){
		free(DATA[i]);
	}
	return 0;
}
