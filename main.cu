#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h> // rand, srand
#include <time.h> 
#include <omp.h>

#define BLOCK_SIZE 16

// Midway point of fire
unsigned int middle       = 0; 

// number of phases, and probability of igniting
int fireSpreadPhases      = 0;
int fireSpreadProbability = 0;

__global__ void spreadFireGPU(int *a, int *b, int N)
{
	// calculate the row and column index of the element
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;


	b[row * N + col] = 1;

	if(a[row*N+col] == 0){			b[row*N+col] = 0; } // do nothing
	else if(a[row * N + col] == 1){ b[row*N+col] = 1; }
	else if(a[row * N + col] == 2){ b[row*N+col] = 2; }
}



void spreadFireCPU(int *a, int *b, int N)
{
	srand(time(NULL));

	for(int i = 0; i < fireSpreadPhases; i++)
	{
		for(int row = 0; row < N; row++)
		{
			for(int col = 0; col < N; col++)
			{
				int index = row * N + col;

				switch(a[index])
				{
					case 0:
					{
						b[index] = 0;
						break;
					}
					case 1: 
					{
						// check the neighbourhood - cellular automata
						// if a neighbouring element has a tree that is on fire
						// then randomly see if the element itself is going to be ignited
						if(a[index + N] == 2 || a[index - N] == 2 ||
						   a[index + 1] == 2 || a[index - 1] == 2 ||
						   a[(index + N) - 1] == 2 || a[(index + N) + 1] == 2 ||
						   a[(index - N) - 1] == 2 || a[(index - N) + 1] == 2)
						{
							if(rand() % 100 < fireSpreadProbability) 
								b[index] = 2;
							else 
								b[index] = 1;
						}
						else 
							b[index] = 1;

						break;
					}
					case 2: 
					{
						// if a tree is on fire, there is a 30% chance that it will be burnt down completely
						if(rand() % 100 < 30)
							b[index] = 0;
						else b[index] = 2;
						break;
					}
				}
			}
		}
		// for each simulation/phase, copy the events over to a
		a = b;
	}
}

int main()
{
	// Number of elements (NxN)
	int N    = 20;

	// host memory pointers
	int *a_h = NULL;
	int *b_h = NULL;

	// grid and block size
	int threads = 128;

	// optimal number of blocks
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	int maxBlocks = deviceProp.maxThreadsPerMultiProcessor/threads*deviceProp.multiProcessorCount;

	dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
	dim3 grid(maxBlocks, 1, 1);

	// number of byes in the array
	size_t memSize = (N*N) * sizeof(int); 

	// allocate memory on the host
	a_h = (int*) malloc(memSize);
	b_h = (int*) malloc(memSize);

	// device memory pointers
	int *a_d = NULL;
	int *b_d = NULL;

	// allocate memory on the device
	cudaMalloc((void**)&a_d, memSize);
	cudaMalloc((void**)&b_d, memSize);

	// Find midway point in N to determine when the fire starts
	// Half way across the total length of the row
	// And the column is half the total amount of columns
	if(N % 2 == 0)
		middle = N / 2; 
	else
	{
		// In the case that N cannot be divided by 2 without a remainder
		middle = N;

		while (middle % 2 != 0)
			middle += 1;  

		middle = middle / 2;
	}

	// STARTUP - number of simultations and the probability of fire spreading
	printf("\nFires in forests spread over time; please determine the amount of phases (or turns) "
		"that there will be for this session ");
	scanf("%i", &fireSpreadPhases);

	printf("\nEnter the probability of fire spreading from an ignited tree to those adjacent to it (0 - 100): ");
	scanf("%i", &fireSpreadProbability);

	// middle of N - where the fire starts - middle = center of rows and cols
	printf("\nThe starting point of the fire is at row: %i col: %i", middle, middle);
	printf("\nThe total number of elements are:         %i \n" , N);

	// Load cpu array with numbers
	// The central point will be given a value of 2 -- tree is on fire
	// This is to determine the starting point for the simulation.
	for (int i = 0; i < N; i++)
	{
		// first row is all  empty; as is last
		if(i == 0 || i == N - 1)
			for (int j = 0; j < N; j++)

				a_h[i * N + j] = 0;
		else
		{
			for (int j = 0; j < N; j++)

				if(j == 0 || j == N - 1)

					a_h[i * N + j] = 0; // empty space (boundary to avoid index/memory error)
				else if(i == middle && j == middle)

					a_h[i * N + j] = 2; // tree ignited
				else 
					a_h[i * N + j] = 1;
		}
	}

	// Copy initialisation data above to the device
	cudaMemcpy(a_d, a_h, memSize, cudaMemcpyHostToDevice);
	cudaMemcpy(b_d, b_h, memSize, cudaMemcpyHostToDevice);


	// Time variables
	float cpuTime;

	float startCPU = omp_get_wtime();

	printf("\nMeasuring CPU execution time...\n");

	// perform cpu computation to determine spread of fire
	spreadFireCPU(a_h, b_h, N);

	float endCPU = omp_get_wtime();
	cpuTime = (endCPU - startCPU) *1000;

	printf("The CPU implementation speed is: %f ms \n\n", cpuTime);

	// print out to be sure it is correct
	for(int i = 0; i < N; i ++)
	{
		for (int j = 0; j < N; j++)
		{
			printf(" %i ", b_h[i * N + j]);
		}
		printf("\n");
    }

	// STARTING GPU Implementation
	printf("Measuring GPU execution time...\n");
	
	// get number of elements

	spreadFireGPU<<<grid, block>>>(a_d, b_d, N);

	// collect results from GPU
	cudaMemcpy(b_h, b_d, memSize, cudaMemcpyDeviceToHost);

	//for(int i = 0; i < N; i ++)
	//{
	//	for (int j = 0; j < N; j++)
	//	{
	//		printf(" %i ", b_h[i * N + j]);
	//	}
	//	printf("\n");
    //}

	// free device memory
	cudaFree(a_d);
	cudaFree(b_d);

	// free up some memory
	free(a_h);
	free(b_h);

	return 0;
}