#include <stdio.h>

#define arraySize 1000

__global__ void vector_add(int * C, const int * A, const int * B, const int size)
{
   int item = (blockIdx.x * blockDim.x) + threadIdx.x;
   
   if ( item < size )
   {
      C[item] = A[item] + B[item];
   }
}

__global__ void use_local_memory_GPU(float in)
{
    float f;    // variable "f" is in local memory and private to each thread
    f = in;     // parameter "in" is in local memory and private to each thread
    // ... real code would presumably do other stuff here ... 
}

/**********************
 * using global memory *
 **********************/

// a __global__ function runs on the GPU & can be called from host
__global__ void use_global_memory_GPU(float *array)
{
    // "array" is a pointer into global memory on the device
    array[threadIdx.x] = 2.0f * (float) threadIdx.x;
}

int main()
{
    int a[arraySize];
    int b[arraySize];
    int c[arraySize];

    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;

    // fill the arrays 'a' and 'b' on the CPU
    for( int i = 0 ; i < arraySize ; i++ ) {
	a[i] = i;
	b[i] = i;
    }

    // Add vectors in parallel.
    // Allocate GPU buffers for three vectors (two input, one output)
    cudaMalloc((void**)&dev_c, arraySize * sizeof(int));
    cudaMalloc((void**)&dev_a, arraySize * sizeof(int));
    cudaMalloc((void**)&dev_b, arraySize * sizeof(int));

    // copy the arrays 'a' and 'b' to the GPU
    cudaMemcpy(dev_a, a, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, arraySize * sizeof(int), cudaMemcpyHostToDevice);

    vector_add<<<1, arraySize>>>(dev_c, dev_a, dev_b, arraySize);
    cudaDeviceSynchronize();

    // copy the array 'c' back from the GPU to the CPU
    cudaMemcpy(c, dev_c, arraySize * sizeof(int), cudaMemcpyDeviceToHost);

    // display the results
    for( int i = 0 ; i < arraySize ; i++ ) {
	printf( "%d + %d = %d\n", a[i], b[i], c[i] );
    }

    // free the memory allocated on the GPU
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return 0;
}
