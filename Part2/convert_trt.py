import torch
from torch2trt import torch2trt
from mlp_eigval import EigenvalueMLP  # assuming you move your model class into mlp_benchmark.py
import time

# Setup
device = torch.device("cuda")
matrix_size = 4
input_dim = matrix_size * matrix_size

# Load model
model = EigenvalueMLP(input_dim).to(device)
model.load_state_dict(torch.load("mlp_eigenvalue_model.pth"))
model.eval()

# Example input for conversion
x_dummy = torch.randn(100, input_dim).to(device)

# Convert to TensorRT
model_trt = torch2trt(model, [x_dummy], fp16_mode=True)

# Save TRT model
torch.save(model_trt.state_dict(), "mlp_eigenvalue_model_trt.pth")
print("TensorRT model saved.")

# ###### Inference speed check between torch and tensorRT models ######
model.eval()
model_trt.eval()

x_test = torch.randn(100, input_dim).to(device)

with torch.no_grad():
    start = time.time()
    output_base = model(x_test)
    torch.cuda.synchronize() # wait for all GPU blocks/threads to complete
    base_time = time.time() - start

    start = time.time()
    output_trt= model_trt(x_test)
    torch.cuda.synchronize() # wait for all GPU blocks/threads to complete
    trt_time = time.time() - start

print(f"PyTorch inference time: {base_time:.4f} sec")
print(f"TensorRT inference time: {trt_time:.4f} sec")