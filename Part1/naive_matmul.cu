#include <iostream>
#include <chrono>

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
typedef struct {
    int width;
    int height;
    float* elements;
} Matrix;

#define BLOCK_SIZE 16

// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
    // Each thread computes one element of C
    // by accumulating results into Cvalue
    float Cvalue = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    for (int e = 0; e < A.width; ++e)
        Cvalue += A.elements[row * A.width + e]
                * B.elements[e * B.width + col];
    C.elements[row * C.width + col] = Cvalue;
}

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(const Matrix A, const Matrix B, Matrix C)
{
    // Load A and B to device memory
    Matrix d_A;
    d_A.width = A.width; 
    d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size); // Allocate device memory
    cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);
    
    Matrix d_B;
    d_B.width = B.width; 
    d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);

    // Allocate C in device memory
    Matrix d_C;
    d_C.width = C.width; 
    d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    cudaMalloc(&d_C.elements, size);

    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

    // Read C from device memory
    cudaMemcpy(C.elements, d_C.elements, size,
               cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}

int main() {
    // Set the size of the matrices
    int array_sizes[5] = {128, 512, 1024, 2048};

    for (int i = 0; i < 4; ++i) {
        const int N = array_sizes[i];

        // Allocate and initialize A, B, and C on host
        Matrix A = {N, N, new float[N * N]};
        Matrix B = {N, N, new float[N * N]};
        Matrix C = {N, N, new float[N * N]};

        // Fill A and B with random values
        for (int i = 0; i < N * N; ++i) {
            A.elements[i] = static_cast<float>(rand()) / RAND_MAX;
            B.elements[i] = static_cast<float>(rand()) / RAND_MAX;
        }

        auto start = std::chrono::high_resolution_clock::now();
        MatMul(A, B, C);
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> elapsed = end - start;
        std::cout << "Matrix size: " << N << "x" << N 
                << " | Time: " << elapsed.count() << " ms\n";

        // Free host memory
        delete[] A.elements;
        delete[] B.elements;
        delete[] C.elements;
    }

    return 0;
}