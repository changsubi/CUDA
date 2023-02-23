# CUDA
Nvidia GPU using CUDA Core.   
How to use CUDA Programming.   
![2D_Convolution_Animation](https://user-images.githubusercontent.com/100255173/219933368-3171935e-c6e5-4541-9925-beca5450abb1.gif)
## 1. GPU Programming: Step by Step
• Setup inputs on the host (CPU-accessible memory)   
• Allocate memory for outputs on the host CPU   
• Allocate memory for inputs on the GPU   
• Allocate memory for outputs on the GPU   
• Copy inputs from host to GPU (slow)   
• Start GPU kernel (function that executes on gpu – fast!)   
• Copy output from GPU to host (slow)   
The Kernel
• This is our “parallel” function   
• Given to each thread   
• Simple example, implementation:   
![image](https://user-images.githubusercontent.com/100255173/220850178-69303f71-f43f-47e9-a4e6-676f555d2bd1.png)
