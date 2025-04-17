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

root@ubuntu:/workspace/Part2# python3 convert_trt.py 
/workspace/Part2/convert_trt.py:13: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  model.load_state_dict(torch.load("mlp_eigenvalue_model.pth"))
TensorRT model saved.
[04/17/2025-16:19:54] [TRT] [E] IExecutionContext::setInputShape: Error Code 3: API Usage Error (Parameter check failed, condition: engineDims.d[i] == dims.d[i]. Static dimension mismatch while setting input shape for input_0. Set dimensions are [100,16]. Expected dimensions are [1,16].)
[04/17/2025-16:19:54] [TRT] [W] Using default stream in enqueueV3() may lead to performance issues due to additional calls to cudaStreamSynchronize() by TensorRT to ensure correct synchronization. Please use non-default stream instead.
PyTorch inference time: 0.0102 sec
TensorRT inference time: 0.0030 sec
```



###  Lecture Notes
number representation highly affects GPU performance gains (10^-7 error in output). From lecture Yogesh. FP32 --> FP16 --> Int8 (quantization) gives less range, better precision --> more range, less precision \

BF16: 8 bit exponents, but increased range with BF16
- 4x rediction yields 16x improvement in TOPS

small gradients become 0, which is bad if losse become small (vanishing gradient)

linearly seperable matrix much better (task changes, not model changes)

tiled matmult
transformer engine (for transfoemr hardware optimization/specialization)

quantization question: the reduction in "range" improves generality, but doesn't decrease output model accuarcy significantly.