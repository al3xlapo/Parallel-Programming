#include "mpiFunctions.h"
#include "mpi.h"

//find the neighbors of the process
int *findNeighProcs(MPI_Comm comm, int procs, int rank){
	
    int *neighProcs = malloc(8*sizeof(int));
	int coords[2]; //coords[0] is the "row" or 'y'-vertical coordinate, and coord[1] is the "column" or 'x'-horizontal coordinate
	
	//first find the 4 neighbors with whom this process will exhange rows (N, S) and columns(W, E)
	MPI_Cart_shift(comm, 0, 1, &neighProcs[N], &neighProcs[S]);
	MPI_Cart_shift(comm, 1, 1, &neighProcs[W], &neighProcs[E]);
	MPI_Cart_coords(comm, rank, 2, coords); //find this process's coordinates
	//then find the diagonal neighboring process by altering this process's coords
	coords[0]-=1;
    coords[1]+=1;
	MPI_Cart_rank(comm, coords, &neighProcs[NE]);
    coords[1]-=2;
	MPI_Cart_rank(comm, coords, &neighProcs[NW]);
    coords[0]+=2;
	MPI_Cart_rank(comm, coords, &neighProcs[SW]);
    coords[1]+=2;
	MPI_Cart_rank(comm, coords, &neighProcs[SE]); 
	
    
    /*MPI_Cart_coords(comm,neighProcs[N],2,coords);
    coords[1]+=1;
    MPI_Cart_rank(comm,coords,&neighProcs[NE]);
    coords[1]-=2;
    MPI_Cart_rank(comm,coords,&neighProcs[NW]);
    coords[0]+=2;
    MPI_Cart_rank(comm,coords,&neighProcs[SW]);
    coords[1]+=2;
    MPI_Cart_rank(comm,coords,&neighProcs[SE]); */
	
    return neighProcs;
}

//the outter rows and columns of the subarrrays are reserved for the halo//
void sendHalo(MPI_Comm comm, MPI_Datatype ROW, MPI_Datatype COLUMN, int** image, int rows, int cols, int* nProcs, MPI_Request* req){
	MPI_Isend(&image[1][1], 1, ROW, nProcs[N], S, comm, &req[N]);
	MPI_Isend(&image[1][cols-2], 1, MPI_INT, nProcs[NE], SW, comm, &req[NE]);
	MPI_Isend(&image[1][cols-2], 1, COLUMN, nProcs[E], W, comm, &req[E]);
	MPI_Isend(&image[rows-2][cols-2], 1, MPI_INT, nProcs[SE], NW, comm, &req[SE]);
	MPI_Isend(&image[rows-2][1], 1, ROW, nProcs[S], N, comm, &req[S]);
	MPI_Isend(&image[rows-2][1], 1, MPI_INT, nProcs[SW], NE, comm, &req[SW]);
	MPI_Isend(&image[1][1], 1, COLUMN, nProcs[W], E, comm, &req[W]);
	MPI_Isend(&image[1][1], 1, MPI_INT, nProcs[NW], SE, comm, &req[NW]);
}

/*void sendHalo(MPI_Comm comm, MPI_Datatype ROW, MPI_Datatype COLUMN, int** image, int rows, int cols, int* nProcs, MPI_Request* req){
	MPI_Isend(&image[0][0], 1, ROW, nProcs[N], S, comm, &req[N]);
	MPI_Isend(&image[0][cols-1], 1, MPI_INT, nProcs[NE], SW, comm, &req[NE]);
	MPI_Isend(&image[0][cols-1], 1, COLUMN, nProcs[E], W, comm, &req[E]);
	MPI_Isend(&image[rows-1][cols-1], 1, MPI_INT, nProcs[SE], NW, comm, &req[SE]);
	MPI_Isend(&image[rows-1][0], 1, ROW, nProcs[S], N, comm, &req[S]);
	MPI_Isend(&image[rows-1][0], 1, MPI_INT, nProcs[SW], NE, comm, &req[SW]);
	MPI_Isend(&image[0][0], 1, COLUMN, nProcs[W], E, comm, &req[W]);
	MPI_Isend(&image[0][0], 1, MPI_INT, nProcs[NW], SE, comm, &req[NW]);
} */

void recvHalo(MPI_Comm comm, MPI_Datatype ROW, MPI_Datatype COLUMN, int** image, int rows, int cols, int* nProcs, MPI_Request* req){
	MPI_Irecv(&image[0][1], 1, ROW, nProcs[N], N, comm, &req[S]);
	MPI_Irecv(&image[0][cols-1], 1, MPI_INT, nProcs[NE], NE, comm, &req[SW]);
	MPI_Irecv(&image[1][cols-1], 1, COLUMN, nProcs[E], E, comm, &req[W]);
	MPI_Irecv(&image[rows-1][cols-1], 1, MPI_INT, nProcs[SE], SE, comm, &req[NW]);
	MPI_Irecv(&image[rows-1][1], 1, ROW, nProcs[S], S, comm, &req[N]);
	MPI_Irecv(&image[rows-1][0], 1, MPI_INT, nProcs[SW], SW, comm, &req[NE]);
	MPI_Irecv(&image[1][0], 1, COLUMN, nProcs[W], W, comm, &req[E]);
	MPI_Irecv(&image[0][0], 1, MPI_INT, nProcs[NW], NW, comm, &req[SE]);
}