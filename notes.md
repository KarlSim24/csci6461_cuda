# Project Notes

### Part 1: Jetson Orin Nano: basic CUDA kernel experiment vs PyTorch CPU+GPU
    - explain how the cuda kernel works
    - use the shmem example as well
### Part 2: Jetson Orin Nano: Simple MLP in pytorch vs TensorRT+cuDNN
    - explain what these do (opimizations) from a memory/architecture POV

Kernels: C++ functions that when called are executed N times in parallel by N different CUDA threads
- Thread: execution of a kernel on a single piece of data
    - gets mapped to single CUDA core when kernel is launched
- Thread Block: threads grouped into blocks
- Grid: set of blocks (1D, 2D, 3D)

![CUDA Architecture Overview](/home/karlsimon/csci6461/final/csci6461_cuda/naive_matmul.png)

Results (on GPU SERVER):
```
(cuda_project) karlsimon@fdcl1:~/csci6461/final/csci6461_cuda$ ./matmul 
Matrix size: 128x128 | Time: 283.782 ms
Matrix size: 512x512 | Time: 1.14644 ms
Matrix size: 1024x1024 | Time: 3.72943 ms
Matrix size: 2048x2048 | Time: 16.6523 ms

(cuda_project) karlsimon@fdcl1:~/csci6461/final/csci6461_cuda$ python pytorch_matmul.py 
Benchmarking PyTorch matmul on cpu
Size: 128×128 | Avg Time: 0.060 ms
Size: 512×512 | Avg Time: 0.199 ms
Size: 1024×1024 | Avg Time: 1.061 ms
Size: 2048×2048 | Avg Time: 8.819 ms
Benchmarking PyTorch matmul on cuda
Size: 128×128 | Avg Time: 0.030 ms
Size: 512×512 | Avg Time: 0.037 ms
Size: 1024×1024 | Avg Time: 0.139 ms
Size: 2048×2048 | Avg Time: 0.993 ms
```

Results on Jetson Container with TensorRT/cuDNN installed (MAXN SUPER mode)


```
root@ubuntu:/workspace# nvcc naive_matmul.cu -o matmul
root@ubuntu:/workspace# ./matmul
Matrix size: 128x128 | Time: 155.183 ms
Matrix size: 512x512 | Time: 11.5607 ms
Matrix size: 1024x1024 | Time: 66.6798 ms
Matrix size: 2048x2048 | Time: 199.22 ms

root@ubuntu:/workspace# python3 pytorch_matmul.py 
Benchmarking PyTorch matmul on cpu
Size: 128×128 | Avg Time: 0.242 ms
Size: 512×512 | Avg Time: 7.368 ms
Size: 1024×1024 | Avg Time: 36.232 ms
Size: 2048×2048 | Avg Time: 177.358 ms
Benchmarking PyTorch matmul on cuda
Size: 128×128 | Avg Time: 0.191 ms
Size: 512×512 | Avg Time: 0.873 ms
Size: 1024×1024 | Avg Time: 8.737 ms
Size: 2048×2048 | Avg Time: 20.938 ms
```