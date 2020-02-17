#include "convolution.h"
#include <math.h>
#include <unistd.h>

int main(int argc, char **argv){
	
	int **I0, **I1, i,j, k=0, colN, rowN, cores, coreTotal, rank;
	float** h;
	char *filter = "121242121", c, *image=NULL; //filter for convolution as a string to make it into the array "h"
	srand(time(NULL));
	
	while((c = getopt(argc, argv,"i:c:r:")) != -1){
        	switch(c)
        	{
            	case 'c': //number of columns of pixels
               		colN = atoi(optarg);
               		break;
				case 'r':  //number of rows of pixels
               		rowN = atoi(optarg);
               		break;
				case 'i':
					image = optarg; //read the image to be convoluted
					break;
        	}
	}
	
	if(colN<=0 || rowN<=0){
		printf("Non Positive image dimensions\n");
		return -1;
	}
		

	//
	//allocate the two arrays and the filter we're working with at any point in the iterative convolution, contiguously
	I0 = allocImage(rowN, colN);
	I1 = allocImage(rowN, colN);
	//allocate the filter so that memory is contiguous, necessary for ROW and COLUMN datatypes
	h = (float**)malloc(3*sizeof(float*));  //allocate pointers
	float* data = (float*)malloc(3*3*sizeof(float));
	for(i=0; i<3; i++)
    		h[i] = &(data[i*3]);
	for(i=0; i<3; i++) 
		for(j=0; j<3; j++){
			h[i][j] = ((float)(filter[k] - '0')) / 16;
			k++;
		}

	//initialize MPI
	int mpiSupp;
	MPI_Init_thread(&argc,&argv, MPI_THREAD_FUNNELED, &mpiSupp);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank); //this process's rank
	MPI_Comm_size(MPI_COMM_WORLD,&coreTotal); //total number of processes
	//periodic for ease while computing the outter elements of the array
	int dims[2], periodic[2] = {1,1}, rowP, colP, reorder=1;
	//define process topology
	MPI_Comm cartComm; //we'll be using cartesian topology for the processes
	cores = sqrt(coreTotal);
	dims[0] = cores;
	dims[1] = cores;
	if(rowN%cores != 0 || colN%cores != 0){
		printf("ERROR, rowsMODprocs or colsMODprocs gives !=0\n");
		MPI_Finalize();
		exit(1);
	}
	MPI_Cart_create(MPI_COMM_WORLD, DIMS, dims, periodic, reorder, &cartComm);
	//
	
	//allocate input array
	if(rank == 0){
		if(image != NULL){
			//make an array out of the input image
			I1 = inputImg(I1, rowN, colN, rank, cartComm, image);
		}
		else{
			//fill the array with random numbers
			for(i=0; i<rowN; i++){
				for(j=0; j<colN; j++){
					I1[i][j]=rand()%255;
					I0[i][j] = 0;
				}
			}
		}
	}
	//
	
	if(coreTotal <= 1){
		//sequential convolution
		I0 = serialConv(&I1, &I0, h, rowN, colN);
		outputImg(I0, rowN, colN);
	}
	else{
		//parallel convolution, utilize 4 or more cores
		int* nProcs = findNeighProcs(cartComm, cores, rank);
		I0 = parallelConv(cartComm, I1, I0, h, rowN, colN, cores, nProcs);
		if(rank==0)
			outputImg(I0, rowN, colN);
	}
	
	
		
	free(I1);
	free(h);
	free(I0);
	MPI_Comm_free(&cartComm);
	MPI_Finalize();
	
}
