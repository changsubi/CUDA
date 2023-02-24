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
<img src="https://user-images.githubusercontent.com/100255173/220858875-d7d3b4c7-734b-4558-9407-cbb6c67c18a1.png" width="200px" height="250px"></img>

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
• Registers   
• Local Memory   
• Global Memory   
• Shared Memory   
• L1/L2/L3 Cache   
• Texture Memory   
• Constant Memory (Read-Only Cache)   

**Registers**  
In the case of CPU, each core has only a few hundred registers. On the other hand, in the case of GPU, the register size is quite large.    
Usually, one SM has about 65,000 registers. Assuming a 32-Bit Register, the Register size is approximately 256 KB (4 Byte X 65,000).   
Registers are fast on-chip memories that are used to store operands for the operations executed by the computing cores.   
Registers are local to a thread, and each thread has exclusive access to its own registers: values in registers cannot be accessed by other threads, even from the same block, and are not available for the host. Registers are also not permanent, therefore data stored in registers is only available during the execution of a thread.

**Local Memory**   
If the register usage is too large, the value of a specific register (less accessible) is stored in local memory and the register is used for other purposes. (register spilling)   
When compiling CUDA code using NVCC, the number of registers used per thread can be set. At this time, if the register usage is set small, local memory is used. However, local memory is very slow(Similarly Global Memory).   
It is better to let the compiler know how to use local memory. Local memory is automatically coalesced for memory access.








