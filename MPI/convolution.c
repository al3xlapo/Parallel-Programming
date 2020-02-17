#include "convolution.h"


int** allocImage(int rowN,int colN){
	
	//allocate the array so that memory is contiguous, necessary for ROW and COLUMN datatypes
	int** I = (int**)malloc(rowN*sizeof(int*));  //allocate pointers
	int* data = (int*)malloc(rowN*colN*sizeof(int));
	for(int i=0; i<rowN; i++)
    		I[i] = &(data[i*colN]);
	return I;
}

int** callocImage(int rowN,int colN){
	int** I = (int**)calloc(rowN,sizeof(int*));  //allocate pointers
	int* data = (int*)calloc(rowN*colN,sizeof(int));
	for(int i=0; i<rowN; i++)
    		I[i] = &(data[i*colN]);
	return I;
}

int** inputImg(int** I1, int rowN, int colN, int rank, MPI_Comm comm, char* image){
	
	MPI_File mpiFile;
	//MPI_File_open(comm, image, MPI_MODE_RDONLY , MPI_INFO_NULL, &mpiFile);
	//int start[2] = {0,0};
	//int subSizes[2] = {rowN,colN};
    //int bigSizes[2] = {rowN,colN};
    //MPI_Datatype DATA;
    //MPI_Type_create_subarray(2, bigSizes, subSizes, start, MPI_ORDER_C, MPI_INT, &DATA);
    //MPI_Type_commit(&DATA);
	FILE* fp;
	fp = fopen (image,"r+");
	 
    //MPI_File_set_view(mpiFile, 0, MPI_CHAR, DATA, "native", MPI_INFO_NULL);
    /*for(int i = 0; i < rowN; i++){
        for(int j = 0; j < colN; j++){
			MPI_File_read(mpiFile, &(I1[i][j]), 1, MPI_CHAR, MPI_STATUS_IGNORE);
        }
    }*/
	/*for(int i = 0; i < rowN; i++ ){
		MPI_File_read(mpiFile, &(I1[i][0]), colN, MPI_CHAR, MPI_STATUS_IGNORE);	
	} */
	//MPI_File_close(&mpiFile);
	//MPI_Type_free(&DATA);
	
	//
	fseek(fp, 0, SEEK_SET );
	for(int i = 0; i < rowN; i++){
        for(int j = 0; j < colN; j++){
			fread(&I1[i][j], sizeof(char), 1, fp);
		}
	} 
	//
    fclose(fp);
	return I1;
}

void outputImg(int** I0, int rowN, int colN){
	FILE* fp;
	fp = fopen ("output.txt","w+");
	fseek(fp, 0, SEEK_SET );
	for(int i = 0; i < rowN; i++){
        for(int j = 0; j < colN; j++){
			fprintf(fp, "%d ", I0[i][j]);
		}
		fprintf(fp, "\n");
	} 
	
	
}
//check to see if the convoluted image is the same as the previous
int isAltered(int **IX,int **IY, int rowN, int colN){
    for(int i = 0; i < rowN; i++){
        for(int j = 0; j < colN; j++){
            if(IX[i][j] != IY[i][j])  return 1;
        }
    }
    return 0; //it's unaltered
}


//serial convolution
int** serialConv(int*** I1, int*** I0, float** h, int rowN, int colN){

	int i,j, indexi, indexj, **auxPtr, stableRounds=0, stable=0;
	double start_t, end_t, passedTime, finalTime;
	start_t = MPI_Wtime();
	for(int ll=0; ll<50; ll++){
		memset(I0[0][0], 0, rowN*colN*sizeof((*I0)[0][0]));
		for(i=0; i<rowN; i++){
			for(j=0; j<colN; j++){
				int p,q;
				for(p=-1; p<=1; p++){
					for(q=-1; q<=1; q++){
						if(i-p < 0)
							indexi = rowN - 1;
						else if(i-p >=rowN )
							indexi = 0;
						else
							indexi = i-p;
						
						if(j-q < 0)
							indexj = colN - 1;
						else if(j-q >= colN )
							indexj = 0;
						else
							indexj = j-q;
						
						(*I0)[i][j] += (float)(*I1)[indexi][indexj]*h[p+1][q+1];
							
					}
				}
			}
		}
		auxPtr = *I1;
		*I1 = *I0;
		*I0 = auxPtr;
	}
	
	end_t = MPI_Wtime();
    passedTime = end_t - start_t;
    printf("Time passed: %f\n", passedTime);
	
	return *I0;
}


int** parallelConv(MPI_Comm comm, int** I1, int** I0, float** h, int rowN, int colN, int procs, int* nProcs){ 
	//procs is the number of processes in one dimension(the grid has #procs*procs processes)
	int i, j, rank, np, baseR, extraR, baseC, extraC, myProc[2], stableRounds=0, stable=0;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &np);
		
	//set up the matrixes needed for parallel convolution
	int pRows[procs*procs];
	int pCols[procs*procs];
	int firstR[procs*procs];
	int firstC[procs*procs];
	
	baseR = rowN/procs; //minimum number of rows each process will have
	extraR = rowN%procs; //extra rows, they will be distributed to the first extraR process rows
	baseC = colN/procs; //same for columns but horizontally
	extraC = colN%procs;
	baseR += 2; //add 2 rows and 2 columns to make space for other proesses' halos
	baseC += 2;
	//define the number of rows each process will handle
	for(i=0; i<extraR; i++) 
		for(j=0; j<procs; j++)
			pRows[i*procs+j] = baseR + 1;
	for(i = extraR; i < procs; i++) 
		for(j=0; j<procs; j++)
			pRows[i*procs+j] = baseR;
	//define the number of columns each process will handle
	for(j=0; j<extraC; j++)
		for(i=0; i<procs; i++)
			pCols[i*procs+j] = baseC + 1;
	for(j=extraC; j<procs; j++)
		for(i=0; i<procs; i++)
			pCols[i*procs+j] = baseC;
	//define the first row of the submatrix(subimage) each process will handle
	for (i=0; i<procs; i++) {
		if(i==0){
			for(j=0; j<procs; j++)
				firstR[i*procs+j] = 0;
		}
		else{
			for(j=0; j<procs; j++)
				firstR[i*procs+j] = firstR[(i-1)*procs+j] + pRows[(i-1)*procs+j];
		}
	} 
	//define the first column of the submatrix(subimage) each process will handle
	for (j=0; j<procs; j++) {
		if(j==0){
			for(i=0; i<procs; i++)
				firstC[i*procs+j] = 0;
		}
		else{
			for(i=0; i<procs; i++)
				firstC[i*procs+j] = firstC[i*procs+j-1] + pCols[i*procs+j-1];
		}
	}
	//find where this process is located on the process grid
	for(i=0; i<procs; i++){
		for(j=0; j<procs; j++){
			if( (i*procs+j) == rank ){
				myProc[0] = i;
				myProc[1] = j;
			}
		}
	}
	
	//define datatypes
	int wholeDims[2] = {rowN, colN}; //dimensions of the whole array-image
	int subDims[2] = {pRows[rank]-2, pCols[rank]-2}; //dimensions of the subarray this process will handle
	int starts[2] = {0,0};
	MPI_Datatype ROW, COLUMN, SOSOIMG, SUBIMG;
	MPI_Type_contiguous(pCols[rank]-2, MPI_INT, &ROW); //row datatype
	MPI_Type_commit(&ROW);
	MPI_Type_vector(pRows[rank]-2, 1, pCols[rank], MPI_INT, &COLUMN); //column datatype
	MPI_Type_commit(&COLUMN);
	//SOSOIMG is the subarray block datatype, but we'll have to specify a set amount of size between the blocks to properly scatter them
	MPI_Type_create_subarray(DIMS, wholeDims, subDims, starts, MPI_ORDER_C, MPI_INT, &SOSOIMG); 
	//so
	MPI_Type_create_resized(SOSOIMG, 0, (pCols[rank]-2)*sizeof(int), &SUBIMG);
	MPI_Type_commit(&SUBIMG);
	
	//scatter the subarrays
	int sendCount[procs*procs];
	int displace[procs*procs];
	
	int* wholePtr = NULL;
	if(rank==0) 
		wholePtr = &(I1[0][0]);
	int** subArr = allocImage(pRows[rank]-2, pCols[rank]-2);
	int** subTarg = callocImage(pRows[rank]-2, pCols[rank]-2);
	
	//find sendCount and displacement parameters for MPI_Scatterv
	if(rank == 0){
		for (i=0; i<procs*procs; i++)
            sendCount[i] = 1;
		int disp=0;
		for (i=0; i<procs; i++) {
			for (j=0; j<procs; j++) {
				displace[i*procs+j] = disp;
				disp++;
				
			}
			disp += (pRows[rank]-3) * procs;
		}
	}
		
	MPI_Scatterv(wholePtr, sendCount, displace, SUBIMG, &(subArr[0][0]), (pRows[rank]-2)*(pCols[rank]-2), MPI_INT, 0, comm);
	//subArr now holds the data each process will compute multiple times
	//haloArr will hold both the data of each array and its halo
	
	int** haloArr  = (int**)calloc(pRows[rank],sizeof(int*));  //allocate pointers
	int* data = (int*)calloc(pRows[rank]*pCols[rank],sizeof(int));
	for(i=0; i<pRows[rank]; i++)
    		haloArr[i] = &(data[i*pCols[rank]]);
		
	
	//subarray pointers to swap after each loop
	int **auxPtr, ***pI0, ***pI1, altered=1; 
	pI0 = &subTarg;
	pI1 = &subArr;
	
	double start_t, end_t, passedTime, finalTime;
	start_t = MPI_Wtime();
	//multiple convolutions loop
	int ll;
	for(ll=0; ll<50; ll++){
		//copy subarray data to new haloArr
		
		for(i=1; i<pRows[rank]-1; i++)
			memcpy(&haloArr[i][1], &subArr[i-1][0], (pCols[rank]-2)*sizeof(int));
		//initialize some variables and reset target subarray
		int reqId, wFlags[8], wDone[8], done=0, tFlag=0;
		for(i=0; i<8; i++){
			wFlags[i]=0;
			wDone[i]=0;
		}
		memset(pI0[0][0], 0, (pRows[rank]-2)*(pCols[rank]-2)*sizeof((*pI0)[0][0]));
		MPI_Request sendReq[8];
		MPI_Request recvReq[8];
		//send and receive halo
		sendHalo(comm, ROW, COLUMN, haloArr, pRows[rank], pCols[rank], nProcs, sendReq);
		recvHalo(comm, ROW, COLUMN, haloArr, pRows[rank], pCols[rank], nProcs, recvReq);

		//compute inner elements
		for(i=1; i<pRows[rank]-3; i++){
			for(j=1; j<pCols[rank]-3; j++){
				for(int p=-1; p<=1; p++){
					for(int q=-1; q<=1; q++){
						subTarg[i][j] += (float)subArr[i-p][j-q]*h[p+1][q+1];
					}
				}
			}	
		}
		//compute outer elements as the required halo parts come one by one to increase speed
		//wFlags will be used to keep track of the parts we have and wDone to see if their equivalent array borders have been computed
		//the loop will break when all have been computed, checking the "done" integer keeping track of the completed "parts"
		while(done<8){
			MPI_Testany(8, recvReq, &reqId, &tFlag, MPI_STATUS_IGNORE);
			if(reqId == N){ ////
				wFlags[N] = 1;
				//compute the upper row
				i=1;
				int barS, barE;
				//if we have the data required and they haven't been convoluted yet, compute this row's edges(NW, NE)
				if(wFlags[NW] && wFlags[W] && !wDone[NW]){
					barS=1;
					wDone[NW]=1;
					done++;
				}
				else
					barS=2;
				if(wFlags[NE] && wFlags[E] && !wDone[NE]){
					barE=1;
					wDone[NE]=1;
					done++;
				}
				else
					barE=2;
				for(j=barS; j<pCols[rank]-barE; j++){
					for(int p=-1; p<=1; p++){
						for(int q=-1; q<=1; q++){
							//in this case we'll have to use an element from the halo
							subTarg[i-1][j-1] += (float)haloArr[i-p][j-q]*h[p+1][q+1];
						}
					}	
				}
				wDone[N]=1;
				done++;
			}
            else if(reqId == NE){ ////
				wFlags[NE] = 1;
				if(wFlags[N] && wFlags[E] && !wDone[NE]){
					for(int p=-1; p<=1; p++){
						for(int q=-1; q<=1; q++){
							subTarg[0][pCols[rank]-3] += (float)haloArr[1-p][pCols[rank]-2-q]*h[p+1][q+1];
						}
					}
					wDone[NE]=1;
					done++;
				}
				
			}
			else if(reqId == E){ ////
				wFlags[E] = 1;
				j=pCols[rank]-2;
				int barS, barE;
				if(wFlags[SE] && wFlags[S] && !wDone[SE]){
					barE=1;
					wDone[SE]=1;
					done++;
				}
				else
					barE=2;
				if(wFlags[NE] && wFlags[N] && !wDone[NE]){
					barS=1;
					wDone[NE]=1;
					done++;
				}
				else
					barS=2;
				for(i=barS; i<pRows[rank]-barE; i++){
					for(int p=-1; p<=1; p++){
						for(int q=-1; q<=1; q++){
							//in this case we'll have to use an element from the halo
							subTarg[i-1][j-1] += (float)haloArr[i-p][j-q]*h[p+1][q+1];
						}
					}	
				}
				wDone[E]=1;
				done++;
			}
			else if(reqId == SE){ ////
				wFlags[SE] = 1;
				if(wFlags[S] && wFlags[E] && !wDone[SE]){
					for(int p=-1; p<=1; p++){
						for(int q=-1; q<=1; q++){
							subTarg[pRows[rank]-3][pCols[rank]-3] += (float)haloArr[pRows[rank]-2-p][pCols[rank]-2-q]*h[p+1][q+1];
						}
					}
					wDone[SE]=1;
					done++;
				}
			}
			else if(reqId == S){ ////
				wFlags[S] = 1;
				i=pRows[rank]-2;
				int barS, barE;
				//if we have the data required and they haven't been convoluted yet, compute this row's edges(NW, NE)
				if(wFlags[SW] && wFlags[W] && !wDone[SW]){
					barS=1;
					wDone[SW]=1;
					done++;
				}
				else
					barS=2;
				if(wFlags[SE] && wFlags[E] && !wDone[SE]){
					barE=1;
					wDone[SE]=1;
					done++;
				}
				else
					barE=2;
				for(j=barS; j<pCols[rank]-barE; j++){
					for(int p=-1; p<=1; p++){
						for(int q=-1; q<=1; q++){
							//in this case we'll have to use an element from the halo
							subTarg[i-1][j-1] += (float)haloArr[i-p][j-q]*h[p+1][q+1];
						}
					}	
				}
				wDone[S]=1;
				done++;
			}
			else if(reqId == SW){ ////
				wFlags[SW] = 1;
				if(wFlags[S] && wFlags[W] && !wDone[SW]){
					for(int p=-1; p<=1; p++){
						for(int q=-1; q<=1; q++){
							subTarg[pRows[rank]-3][0] += (float)haloArr[pRows[rank]-2-p][1-q]*h[p+1][q+1];
						}
					}
					wDone[SW]=1;
					done++;
				}
				
			}
			else if(reqId == W){ ////
				wFlags[W] = 1;
				j=1;
				int barS, barE;
				if(wFlags[SW] && wFlags[S] && !wDone[SW]){
					barE=1;
					wDone[SW]=1;
					done++;
				}
				else
					barE=2;
				if(wFlags[NW] && wFlags[N] && !wDone[NW]){
					barS=1;
					wDone[NW]=1;
					done++;
				}
				else
					barS=2;
				for(i=barS; i<pRows[rank]-barE; i++){
					for(int p=-1; p<=1; p++){
						for(int q=-1; q<=1; q++){
							//in this case we'll have to use an element from the halo
							subTarg[i-1][j-1] += (float)haloArr[i-p][j-q]*h[p+1][q+1];
						}
					}	
				}
				wDone[W]=1;
				done++;
			}
			else if(reqId = NW){ ////
				wFlags[NW] = 1;	
				if(wFlags[N] && wFlags[W] && !wDone[NW]){
					for(int p=-1; p<=1; p++){
						for(int q=-1; q<=1; q++){
							subTarg[0][0] += (float)haloArr[1-p][1-q]*h[p+1][q+1];
						}
					}
					wDone[NW]=1;
					done++;
				}
			}
           
		}
		
		MPI_Waitall(8, sendReq, MPI_STATUS_IGNORE);
		
		//check if it's unaltered
		/*int allStable,staticStable;
        stable = !isAltered(I1,I0,pRows[rank]-2,pCols[rank]-2);
        if(stable == 1){
        	stableRounds++;
        }
        else{
			stableRounds = 0;
        }
        if(ll % 2 == 0){
           	if(stableRounds >= 2){
               	staticStable = 1;
           	}
           	else {
               	staticStable = 0;
           	}
           	MPI_Allreduce(&staticStable,&allStable,1, MPI_INT, MPI_LAND, comm);
           	if(allStable == 1){
				printf("Table stable at round: %d\n", ll);
				break;
			}
        }*/
		auxPtr = *pI1;
		*pI1 = *pI0;
		*pI0 = auxPtr;
	}
	//endloop
	
	end_t = MPI_Wtime();
    passedTime = end_t - start_t;
    MPI_Reduce(&passedTime,&finalTime, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    if(rank == 0){
        printf("Time passed: %f\n",finalTime);
    }
	//process 0 gathers all the convoluted data
	MPI_Gatherv(&subTarg[0][0], (pRows[rank]-2)*(pCols[rank]-2), MPI_INT, &(I0[0][0]), sendCount, displace, SUBIMG, 0, comm);
	
	//free datatypes
	MPI_Type_free(&COLUMN);
	MPI_Type_free(&ROW);
	MPI_Type_free(&SUBIMG);
	
	return I0;
}


