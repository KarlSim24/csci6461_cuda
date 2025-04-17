import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random  
import time

np.random.seed(42)
# Use CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Config
matrix_size = 4 # NxN symmetric matrix
input_dim = matrix_size * matrix_size
epochs = 100
batch_size = 32
learning_rate = 1e-3
num_batches = 10

# Function to create a batch of symmetric matrices and their largest eigenvalue
def generate_batch(batch_size, N):
    A = torch.randn(batch_size, N, N)
    sym_A = 0.5 * (A + A.transpose(1, 2))  # make symmetric
    eigvals = torch.linalg.eigvalsh(sym_A)  # symmetric eig solver
    largest = eigvals[:, -1]  # largest eigenvalue
    return sym_A.view(batch_size, -1), largest

# MLP model
class EigenvalueMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.layers(x).squeeze(1)  # [B]

model = EigenvalueMLP(input_dim).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training
print("Training MLP to learn largest eigenvalue of symmetric matrices...")
start_time = time.time()

for epoch in range(epochs):
    epoch_loss = 0.0
    for _ in range(num_batches):
        output, target = generate_batch(batch_size, matrix_size)
        output, target = output.to(device), target.to(device)

        optimizer.zero_grad()
        pred = model(output)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / num_batches
    print(f"Epoch {epoch:03d} - Avg Loss: {avg_loss:.6f}", flush=True)

end_time = time.time()
print(f"Training completed in {end_time - start_time:.2f} seconds.")

# Save model
torch.save(model.state_dict(), "mlp_eigenvalue_model.pth")
print("Model saved to mlp_eigenvalue_model.pth")

# Test on a single matrix
with torch.no_grad():
    A_test, y_test = generate_batch(1, matrix_size)
    A_test, y_test = A_test.to(device), y_test.to(device)
    pred_test = model(A_test)
    print(f"True largest eigenvalue: {y_test.item():.4f}")
    print(f"Predicted:               {pred_test.item():.4f}")
