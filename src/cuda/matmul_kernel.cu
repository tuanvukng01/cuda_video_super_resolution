#include <torch/extension.h>

__global__ void matmul_kernel(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0;
        for (int i = 0; i < N; ++i) {
            sum += A[row * N + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

void matmul(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& C) {
    int N = A.size(0);
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (N + 15) / 16);
    matmul_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);
}