import torch
import time

sizes = [128, 512, 1024, 2048]
trials = 5

# device is either "cpu" or "cuda"
def benchmark_pytorch_matmul(device):
    
    print("Benchmarking PyTorch matmul on", device)

    for size in sizes:
        times = []
        for _ in range(trials):
            A = torch.randn(size, size, dtype=torch.float32, device=device)
            B = torch.randn(size, size, dtype=torch.float32, device=device)

            torch.matmul(A, B)

            if device == "cuda":
                torch.cuda.synchronize()
                start = time.time()
                torch.matmul(A, B)
                torch.cuda.synchronize() # wait for all kernels to complete before host resumes
                end = time.time()
            else:
                start = time.time()
                torch.matmul(A, B)
                end = time.time()

            times.append(end - start)

        avg_time_ms = sum(times) / trials * 1000
        print(f"Size: {size}×{size} | Avg Time: {avg_time_ms:.3f} ms")

benchmark_pytorch_matmul("cpu")

if torch.cuda.is_available():
    benchmark_pytorch_matmul("cuda")
else:
    print("CUDA not available on this system.")
