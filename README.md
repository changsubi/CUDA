![image](https://user-images.githubusercontent.com/100255173/220853764-85c17842-a01d-4c0c-9e0e-bd6fc6ada62f.png)

# CUDA
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
```
<pre>
<code>
__global__ void
cudaAddVectorsKernel(float *a, float *b, float *c)
{
   //Device an index somehow
   c[index] = a[index] + b[index];
}
</code>
</pre>
```

<pre>
<code>
```java
public class BootSpringBootApplication {
  public static void main(String[] args) {
    System.out.println("Hello, Honeymon");
  }
}
```
</code>
</pre>

```java
public class BootSpringBootApplication {
  public static void main(String[] args) {
    System.out.println("Hello, Honeymon");
  }
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
• GPUs have many Streaming Multiprocessors (SMs)
• Each SM has multiple processors but
only one instruction unit (each thread shares program counter)
• Groups of processors must run the
exact same set of instructions at any
given time with in a single SM
• Think of Device Memory as a RAM for your GPU
• Also refer to it as Global Memory
• Faster than getting memory from the
actual RAM but still have other options
• Will come back to this in future lectures





