# Project Notes

Kernels: C++ functions that when called are executed N times in parallel by N different CUDA threads
- Thread: execution of a kernel on a single piece of data
    - gets mapped to single CUDA core when kernel is launched
- Thread Block: threads grouped into blocks
- Grid: set of blocks (1D, 2D, 3D)

![CUDA Architecture Overview](/home/karlsimon/csci6461/final/csci6461_cuda/naive_matmul.png)