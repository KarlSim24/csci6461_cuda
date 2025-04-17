import torch
from torch2trt import torch2trt
from conv_model import MNISTConvNet
import torchvision.transforms as transforms
import time

# Setup
device = torch.device("cuda")

# Load model
model = MNISTConvNet().to(device)
model.load_state_dict(torch.load("mnist_conv_model.pth"))
model.eval()

# Example input for conversion (batch of 100 MNIST images)
x_dummy = torch.randn(100, 1, 28, 28).to(device)

# Convert to TensorRT
model_trt = torch2trt(model, [x_dummy], fp16_mode=True)

# Save TensorRT model
torch.save(model_trt.state_dict(), "mnist_conv_model_trt.pth")
print("TensorRT model saved to mnist_conv_model_trt.pth")

# Benchmark inference speed
model.eval()
model_trt.eval()
x_test = torch.randn(100, 1, 28, 28).to(device)

# Warm-up
with torch.no_grad():
    _ = model(x_test)
    _ = model_trt(x_test)
    torch.cuda.synchronize()

# Timed runs
with torch.no_grad():
    start = time.time()
    for _ in range(50):
        _ = model(x_test)
    torch.cuda.synchronize()
    base_time = time.time() - start

    start = time.time()
    for _ in range(50):
        _ = model_trt(x_test)
    torch.cuda.synchronize()
    trt_time = time.time() - start

print(f"Avg PyTorch inference time: {base_time / 50:.6f} sec")
print(f"Avg TensorRT inference time: {trt_time / 50:.6f} sec")
