#include <cuda_runtime.h>

__global__ void transposeKernel(const float* A, float* B, int M, int N) {
    
    int j = blockIdx.x * blockDim.x + threadIdx.x; 
    int i = blockIdx.y * blockDim.y + threadIdx.y; 

    if (i < M && j < N) {
        B[j * M + i] = A[i * N + j];
    }
}


extern "C" void solve(const float* A, float* B, int M, int N) {
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (M + blockDim.y - 1) / blockDim.y);

    transposeKernel<<<gridDim, blockDim>>>(A, B, M, N);
    cudaDeviceSynchronize();
}