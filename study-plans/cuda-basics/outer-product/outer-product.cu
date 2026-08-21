#include <cuda_runtime.h>

__global__ void outerProductKernel(const float* a, const float* b, float* C, int M, int N) {
    int j = blockIdx.x * blockDim.x + threadIdx.x; 
    int i = blockIdx.y * blockDim.y + threadIdx.y; 

    if (i < M && j < N) {
        C[i * N + j] = a[i] * b[j];
    }
}


extern "C" void solve(const float* a, const float* b, float* C, int M, int N) {
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (M + blockDim.y - 1) / blockDim.y);

    outerProductKernel<<<gridDim, blockDim>>>(a, b, C, M, N);
    cudaDeviceSynchronize();
}