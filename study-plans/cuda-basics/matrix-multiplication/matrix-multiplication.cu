#include <cuda_runtime.h>

__global__ void matmulKernel(const float* A, const float* B, float* C, int M, int K, int N) {
    int j = blockIdx.x * blockDim.x + threadIdx.x; 
    int i = blockIdx.y * blockDim.y + threadIdx.y; 

    if (i < M && j < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[i * K + k] * B[k * N + j];
        }
        C[i * N + j] = sum;
    }
}


extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (M + blockDim.y - 1) / blockDim.y);

    
    matmulKernel<<<gridDim, blockDim>>>(A, B, C, M, K, N);
    cudaDeviceSynchronize();
}