#include <cuda_runtime.h>

__global__ void matrixAddKernel(const float* A, const float* B, float* C, int M, int N) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // column
    int i = blockIdx.y * blockDim.y + threadIdx.y;  // row

    if (i < M && j < N) {
        int idx = i * N + j;
        C[idx] = A[idx] + B[idx];
    }
}

// A, B, C are device pointers
extern "C" void solve(const float* A, const float* B, float* C, int M, int N) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                        (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrixAddKernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N);
    cudaDeviceSynchronize();
}