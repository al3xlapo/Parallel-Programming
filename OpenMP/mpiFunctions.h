#ifndef mpiFunctions
#define mpiFunctions

#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define DIMS 2
#define N 0
#define NE 1
#define E 2
#define SE 3
#define S 4
#define SW 5
#define W 6
#define NW 7

int *findNeighProcs(MPI_Comm comm, int procs, int my_rank);
void sendHalo(MPI_Comm comm, MPI_Datatype ROW, MPI_Datatype COLUMN, int** image, int rows, int cols, int* nProcs, MPI_Request* req);
void recvHalo(MPI_Comm comm, MPI_Datatype ROW, MPI_Datatype COLUMN, int** image, int rows, int cols, int* nProcs, MPI_Request* req);

#endif