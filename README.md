![image](https://user-images.githubusercontent.com/100255173/220853764-85c17842-a01d-4c0c-9e0e-bd6fc6ada62f.png)

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



