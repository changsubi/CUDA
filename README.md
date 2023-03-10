<img width="711" alt="Logo_and_CUDA" src="https://user-images.githubusercontent.com/100255173/221072615-66330d7e-aa63-4adb-92f4-9a2cf9b8ab19.png">

Nvidia GPU using CUDA Core.   
How to use CUDA Programming.   
![2D_Convolution_Animation](https://user-images.githubusercontent.com/100255173/219933368-3171935e-c6e5-4541-9925-beca5450abb1.gif)
## GPU Programming: Step by Step
• Setup inputs on the host (CPU-accessible memory)   
• Allocate memory for outputs on the host CPU   
• Allocate memory for inputs on the GPU   
• Allocate memory for outputs on the GPU   
• Copy inputs from host to GPU (slow)   
• Start GPU kernel (function that executes on gpu – fast!)   
• Copy output from GPU to host (slow)   
   
## The Kernel   
• This is our “parallel” function   
• Given to each thread   
• Simple example, implementation:   
```cpp
__global__ void
cudaAddVectorsKernel(float *a, float *b, float *c)
{
   //Device an index somehow
   c[index] = a[index] + b[index];
}
```

## Compile It with NVCC   
• CUDA is simply an extension of other bits of code you write!!!!   
• Evident in .cu/.cuh vs .cpp/.hpp distinction   
• .cu/.cuh is compiled by nvcc to produce a .o file   
• Since CUDA 7.0 / 9.0 there’s support by NVCC for most C++11 /
C++14 language features, but make sure to read restrictions for
device code   
• https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#ccplusplus-language-support   
• .cpp/.hpp is compiled by g++ and the .o file from the CUDA code is
simply linked in using a "#include xxx.cuh" call   
• No different from how you link in .o files from normal C++ code   

## Modern GPU: Simplified Version
• GPUs have many **Streaming Multiprocessors** (SMs)  
• SM consists of 8 SP(Scalar Processor)   
• Each SM has multiple processors but only one instruction unit (each thread shares program counter)   
• Groups of processors must run the exact same set of instructions at any given time with in a single SM   
• Think of Device Memory as a RAM for your GPU Also refer to it as **Global Memory** Faster than getting memory from the actual RAM but still have other options   
• The GPU operates (memory reference) in 1SM units (32 threads : 1 Warp)   
• In the case of memory reference with 16 threads, it is called HalfWarp   
• **Scalar Processor**(SP), composed of 4 threads and executed   
• (SM = 8 SPs), (4 threads in 1 SP) : 8*4=32   
• Block: SMs come together to form what is called a block. The number of SMs constituting a block is different for each GPU chip model (usually 16 or 32 SM)   
• Grid: Grid is made up of blocks. Grid can be understood as one GPU chip (usually 65535 blocks)   
• Example> SM per block: 32, block: 65536 → Total available threads: 65536 * 32 * 32   
<img src="https://user-images.githubusercontent.com/100255173/220856882-bf6864cf-834f-4f85-86ac-bb110a2388fe.png" width="200px" height="250px"></img>

## Thread Indexing
<img src="https://user-images.githubusercontent.com/100255173/220861992-b73d9087-f5dd-4bde-a3bc-45cbe826ee2c.png" width="600px" height="400px"></img>
   
In the figure above, each cell means a thread. The orange cell is the thread at index (**threadIdx.x = 0, threadIdx.y = 1**) within Block(**blockIdx.x = 1, blockIdx.y = 1**), and the global index of thread (**x, y**) is (4, 3)   
The global index (**x, y**) is computed as follows:   
**x = threadIdx.x + blockIdx.x * blockDim.x = 0 + 1 * 4 = 4**   
**y = threadIdx.y + blockIdx.y * blockDim.y = 1 + 1 * 2 = 3**   
When placed in a row, the threads corresponding to the orange cells are counted as follows:   
**offset = x + y * blockDim.x * gridDim.x = 4 + 3 * 4 * 2 = 4 + 24 = 28**   
Because GPUs can have a three-dimensional structure, they can of course scale to three dimensions   
```cpp
x = threadIdx.x + blockidx.x * blockDim.x

y = threadIdx.y + blockidx.y * blockDim.y

z = threadIdx.z + blockidx.z * blockDim.z
offset = x + ( y * blockDim.x * gridDim.x ) + ( z * blockDim.x * gridDim.x * blockDim.y * gridDim.y )
```

## GPU Memory
<img src="https://user-images.githubusercontent.com/100255173/220858875-d7d3b4c7-734b-4558-9407-cbb6c67c18a1.png" width="200px" height="250px"></img>   
   
**Registers**  
In the case of CPU, each core has only a few hundred registers. On the other hand, in the case of GPU, the register size is quite large.    
Usually, one SM has about 65,000 registers. Assuming a 32-Bit Register, the Register size is approximately 256 KB (4 Byte X 65,000).   
Registers are fast on-chip memories that are used to store operands for the operations executed by the computing cores.   
Registers are local to a thread, and each thread has exclusive access to its own registers: values in registers cannot be accessed by other threads, even from the same block, and are not available for the host. Registers are also not permanent, therefore data stored in registers is only available during the execution of a thread.

**Local Memory**   
If the register usage is too large, the value of a specific register (less accessible) is stored in local memory and the register is used for other purposes. (register spilling)   
When compiling CUDA code using NVCC, the number of registers used per thread can be set. At this time, if the register usage is set small, local memory is used. However, local memory is very slow(Similarly Global Memory).   
It is better to let the compiler know how to use local memory. Local memory is automatically coalesced for memory access.
```cpp
__global__ void vector_add(const float * A,const float * B, float * C,const int size,const int local_memory_size)
{
   int item = (blockIdx.x * blockDim.x) + threadIdx.x;
   float local_memory[local_memory_size];
   if ( item < size )
   {
      local_memory[0] = A[item];
      local_memory[1] = B[item];
      local_memory[2] = local_memory[0] + local_memory[1];
      C[item] = local_memory[2];
   }
}
```
**Global Memory**  
<img src="https://user-images.githubusercontent.com/100255173/221073189-5450561e-93cf-44af-8c34-43a48c1fa754.png" width="800px" height="200px"></img>    
Global memory is the largest memory on the GPU and the slowest. Currently, you can buy GPUs with dozens of gigabytes of global memory. Access latency is known to be about 400~800 cycles. Accessible from anywhere while coding CUDA, but IO is too slow. So, it should be approached as minimally as possible. You can't not use it though. This is because, in the end, only global memory is connected to the CPU when data is transferred to the CPU.   
While global memory is visible to all threads, remember that global memory is not coherent, and changes made by one thread block may not be available to other thread blocks during the kernel execution. However, all memory operations are finalized when the kernel terminates.

**Memory Coalescing** : The difference in performance between the code that considers memory coalescing and the code that does not occurs 50 to 100%.   
All threads in the warp issue the same memory instruction. When 32 memory requests executed in this way have consecutive memory addresses and access one cache line, it is called coalesced memory access. In this case, a memory request is created by merging the memory instructions of all threads of warp into one. In other words, Warp can handle all memory requests of 32 threads with only one memory access.   
However, since non-coalescing does not enter one cache line, one warp requires two cache accesses. As a result, to process one memory instruction in warp, a total of two memory requests must be generated.
![Untitled](https://user-images.githubusercontent.com/100255173/221075574-4724a164-fa49-49b5-9c01-476c8e9465f6.png)   

**Pinned Memory** : Allocated Host(CPU) memory(RAM) is pageable by default. That is, a page fault operation may occur by the OS to move data from the virtual memory of the host to another physical memory. Just as the L1 cache provides significantly more on-chip memory than physically available, virtual memory provides significantly more memory than physically available.   
On GPUs, data in pageable host memory cannot be safely accessed because the host OS has no control over when the data is physically moved.(it's Global Memory)   
When transferring data from pageable host memory to device memory, the CUDA driver first allocates temporary pinned host memory, copies host data to pinned memory, and then transfers data from pinned memory to device memory.   
![image](https://user-images.githubusercontent.com/100255173/221078197-f43ef132-f0a8-4f8f-90c7-b71a95aaee23.png)   
The CUDA runtime can allocate pinned host memory directly through the following APIs:
```cpp
cudaError_t status = cudaMallocHost((void **)&devPtr, size_t countByte);
if (status != cudaSuccess){
   fprintf(stderr, "Error returned from pinned memory allocation\n");
   exit(1);
}

...

cudaError_t cudaFreeHost(void *ptr); // release pinned memory
```
Allocates as many host memory as count bytes accessible from the device. Because pinned memory can be accessed directly from the device, it has higher read/write bandwidth than pageable memory. However, allocating excessive amounts of pinned memory can reduce the host system's performance by reducing the amount of pageable memory available to the host system that it uses to store virtual memory data.   
   
**Unified Memory** : Devices (GPUs) with compute capability 2.0 or higher support a special addressing mode called **Unified Virtual Addressing (UVA)**. Introduced in CUDA 4.0, UVA is supported on 64-bit Linux systems. Starting with compute capability 3.0 and higher (CUDA 6.0), it is supported under the name Unified Memory on 64-bit Windows systems.   
![image](https://user-images.githubusercontent.com/100255173/221080646-ce340aad-90ad-48ee-bcd1-dfc887b1c0ba.png)   
Prior to the introduction of UVA, pointers to host memory and pointers to device memory had to be managed separately. However, with UVA, the memory space referenced by the pointer is made clear to the application code. Pinned host memory allocated by cudaHostAlloc in UVA has the same host and device pointers. Thus, the returned pointer can be passed directly to kernel functions.   
```cpp
int main(int argc, char** argv)
{
    ...
    
    // using zero-copy memory(can be access host and device) for array A and B
    // allocate zero-copy memory
    CUDA_CHECK(cudaHostAlloc((void**)&h_A, nBytes, cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc((void**)&h_B, nBytes, cudaHostAllocMapped));
 
    // initialize data at host side
    initialData(h_A, nElem);
    initialData(h_B, nElem);
    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);
 
    // need to get the device pointer to the mapped pinned memory. But with UVA support, don't have to this.
    //CUDA_CHECK(cudaHostGetDevicePointer((void**)&d_A, (void*)h_A, 0));
    //CUDA_CHECK(cudaHostGetDevicePointer((void**)&d_B, (void*)h_B, 0));
    
    // add vector at host side for result checks
    sumArraysOnHost(h_A, h_B, hostRef, nElem);
 
    // execute kernel with zero copy memory
    sumArraysZeroCopy<<<grids, blocks>>>(h_A, h_B, d_C, nElem);
    
    ...
}
```

**Deepstream** : In Deepstream, a memory type called 'NvBufSurfaceMemType' is declared in 'nvbufsurface.h'.   
In DeepStream, "default memory" typically refers to the memory that is allocated by the DeepStream framework itself for its internal operations, such as buffering video frames and metadata. This memory is usually allocated on the CPU (host) and is used to communicate data between different components of the pipeline, such as video sources, detectors, and classifiers.   
"Device memory" in DeepStream refers to the memory that is allocated on the GPU and is used by the different processing elements of the pipeline, such as neural network models and image processing kernels. The device memory in DeepStream can include various types of memory, such as global memory, shared memory, and texture memory, which are specific to the CUDA programming model.   
```cpp
 typedef enum
 {
   NVBUF_MEM_DEFAULT,
   NVBUF_MEM_CUDA_PINNED,
   NVBUF_MEM_CUDA_DEVICE,
   NVBUF_MEM_CUDA_UNIFIED,
   NVBUF_MEM_SURFACE_ARRAY,
   NVBUF_MEM_HANDLE,
   NVBUF_MEM_SYSTEM,
 } NvBufSurfaceMemType;
```
If you want to directly access and control the image as a result of Deepstream's analysis, opencv must be used. But, since opencv can be accessed from host memory, the memory type must be set to "NVBUF_MEM_CUDA_UNIFIED".   
Because NVBUF_MEM_DEFAULT is allocated to host memory, but the actual analysis result is not saved. And in the case of NVBUF_MEM_CUDA_PINNED and NVBUF_MEM_CUDA_DEVICE, this is impossible because both point to a pointer to device memory.

```cpp
 ...
 
 gint mem_id = 3;
 #define SET_MEMORY(object, mem_id) g_object_set (G_OBJECT (object), "nvbuf-memory-type", mem_id, NULL);
 ...
 
 for (l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next)
 {
     NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);
     
     /* for cuda device memory we need to use cudamemcpy only using */
     NvBufSurfaceMap(surface, frame_meta->batch_id, 0, NVBUF_MAP_READ_WRITE);
     /* Cache the mapped data for CPU access */
     NvBufSurfaceSyncForCpu(surface, frame_meta->batch_id, 0);
     
     guint height = surface->surfaceList[frame_meta->batch_id].height;
     guint width = surface->surfaceList[frame_meta->batch_id].width;
     
     cv::Mat in_mat = cv::Mat (surface->surfaceList[frame_meta->batch_id].planeParams.height[0], surface->surfaceList[frame_meta->batch_id].planeParams.width[0], CV_8UC4, surface->surfaceList[frame_meta->batch_id].mappedAddr.addr[0], surface->surfaceList[frame_meta->batch_id].planeParams.pitch[0]);
     //Convert RGBA to BGR
     Mat rgba_mat;
     cv::cvtColor(in_mat, rgba_mat, COLOR_RGBA2BGR);
     imwrite(file_name, rgba_mat);
     
     ...
 }
 
 ...
```






