#include <cuda_runtime.h>

__global__ void hadamardKernel(const float* A, const float* B, float* C, int M, int N) {
    int j = blockIdx.x * blockDim.x + threadIdx.x; 
    int i = blockIdx.y * blockDim.y + threadIdx.y; 

    if (i < M && j < N) {
        int idx = i * N + j;
        C[idx] = A[idx] * B[idx];
    }
}


extern "C" void solve(const float* A, const float* B, float* C, int M, int N) {
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (M + blockDim.y - 1) / blockDim.y);

    hadamardKernel<<<gridDim, blockDim>>>(A, B, C, M, N);
    cudaDeviceSynchronize();
}