#include "mpiFunctions.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <mpi.h>

int** allocImage(int rowN,int colN);
int** callocImage(int rowN,int colN);
int isAltered(int **IX,int **IY, int rowN, int colN);
int** serialConv(int*** I1, int*** I0, float** h, int rowN, int colN);
int** parallelConv(MPI_Comm comm, int** I1, int** I0, float** h, int rowN, int colN, int procs, int* nProcs);
int** inputImg(int** I1, int rowN, int colN, int rank, MPI_Comm comm, char* image);
void outputImg(int** I0, int rowN, int colN);